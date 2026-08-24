#!/usr/bin/env Rscript
# Private, schema-validated R adapter for the model-optimization glmnet ridge/elastic-net Cox
# candidates (B2 `glmnet_ridge_core`, B3 `glmnet_enet_core`; optimization-spec.md). Invoked only via
# crc_folfox.optimize.r_adapter.GlmnetAdapter, from a private temporary exchange -- never run
# interactively against restricted data. R fits and predicts; every validation score, lambda
# selection, and comparison stays in Python (tech-stack.md, "Software and the R boundary") -- this
# adapter never selects a lambda or evaluates a metric. Three subcommands:
#
#   --preflight
#       Prints one line of JSON to stdout: {"R", "survival", "glmnet", "jsonlite"} version
#       strings. Exit status 1 on any missing dependency.
#
#   --fit <train_csv> <val_csv> <destination_rds> <response_json> <alpha> <lambda_spec> <seed>
#         <horizons_csv>
#       Fits glmnet(..., family="cox") with a counting-process response on <train_csv>, using
#       Efron ties (cox.ties="efron", matching the project's lifelines convention). <lambda_spec>
#       is either the literal string "path" (compute the frozen nlambda=50/lambda.min.ratio=0.01
#       sequence) or one or more comma-separated numeric values (a previously selected lambda,
#       refit at that exact value only -- the final development-scope refit; or an inner fold
#       forced onto the parent scope's own reference sequence). Predicts absolute risk at every
#       requested horizon in <horizons_csv> for every row of <val_csv> and every lambda in the
#       fitted path, via survival::survfit.coxnet. Saves the fitted glmnet object bundled with the
#       training predictors and response it was fit on -- glmnet's Cox baseline hazard is not
#       stored inside the fitted object itself, so `survfit.coxnet` needs the original training
#       `(x, y)` to reconstruct it even when no refitting occurs -- reloads it, and proves reload
#       determinism before writing the response.
#
#   --predict <bundle_rds> <val_csv> <response_json> <horizons_csv>
#       Loads a bundle saved by --fit and predicts at every lambda it was fit with (typically one,
#       for a final-refit bundle) and every requested horizon, for every row of <val_csv>. No
#       fitting occurs -- this is the evaluation-time path, reusing a frozen bundle exactly as
#       the primary model-training registry does for Python Cox bundles.

.script_path <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
.renv_project <- dirname(dirname(normalizePath(.script_path, mustWork = TRUE)))
.renv_activate <- file.path(.renv_project, "renv", "activate.R")
if (length(.script_path) == 1 && file.exists(.renv_activate)) {
  Sys.setenv(RENV_PROJECT = .renv_project)
  source(.renv_activate)
}

suppressWarnings(suppressMessages(library(jsonlite)))

args <- commandArgs(trailingOnly = TRUE)

fail <- function(message) {
  write(message, stderr())
  quit(status = 1, save = "no")
}

preflight <- function() {
  required <- c("survival", "glmnet", "jsonlite")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    fail(paste("Missing required package(s):", paste(missing, collapse = ", ")))
  }
  suppressWarnings(suppressMessages({
    library(survival)
    library(glmnet)
  }))
  payload <- list(
    R = paste(R.version$major, R.version$minor, sep = "."),
    survival = as.character(utils::packageVersion("survival")),
    glmnet = as.character(utils::packageVersion("glmnet")),
    jsonlite = as.character(utils::packageVersion("jsonlite"))
  )
  cat(toJSON(payload, auto_unbox = TRUE))
}

.read_predictor_matrix <- function(path, predictor_columns) {
  data <- utils::read.csv(path, check.names = FALSE)
  missing_columns <- setdiff(predictor_columns, names(data))
  if (length(missing_columns) > 0) {
    fail(paste(
      "Exchange CSV is missing predictor column(s):", paste(missing_columns, collapse = ", ")
    ))
  }
  as.matrix(data[, predictor_columns, drop = FALSE])
}

# Right-continuous step-function evaluation of one survfit object's curve at `horizons`: S(t) is 1
# strictly before the first event/censoring time, and holds at the value recorded at the most
# recent time <= t elsewhere -- the standard step-function survival-curve convention, extrapolated
# flat beyond the last observed time.
.step_survival <- function(sf, horizons) {
  # `survfit.coxnet$surv` is an (n_times x n_subjects) matrix in general, but R silently drops it
  # to a plain vector when `newx` has exactly one row (the reload-determinism check's own case,
  # since it always re-predicts a single assessment row). Restoring the matrix shape here is what
  # makes this function correct for both call sites without duplicating it.
  surv <- if (is.matrix(sf$surv)) sf$surv else matrix(sf$surv, ncol = 1)
  n_subjects <- ncol(surv)
  out <- matrix(1.0, nrow = n_subjects, ncol = length(horizons))
  for (h in seq_along(horizons)) {
    at_or_before <- which(sf$time <= horizons[h])
    if (length(at_or_before) > 0) {
      out[, h] <- surv[max(at_or_before), ]
    }
  }
  out
}

# Predict every lambda in `fit$lambda` at every requested horizon for every row of `x_val`, using
# the frozen training `(x_train, y_train)` the bundle was fit on. Shared by --fit (against the
# just-fitted object) and --predict (against a reloaded bundle) so the two paths cannot drift.
.predict_all_lambdas <- function(fit, x_train, y_train, x_val, horizons) {
  raw <- survival::survfit(fit, s = fit$lambda, x = x_train, y = y_train, newx = x_val)
  survfits <- if (!is.null(raw$time)) list(raw) else raw
  if (length(survfits) != length(fit$lambda)) {
    fail("survfit did not return one curve set per lambda")
  }
  survival_array <- array(NA_real_, dim = c(length(fit$lambda), nrow(x_val), length(horizons)))
  for (l in seq_along(survfits)) {
    survival_array[l, , ] <- .step_survival(survfits[[l]], horizons)
  }
  if (any(!is.finite(survival_array)) || any(survival_array < 0) || any(survival_array > 1)) {
    fail("glmnet survival prediction is nonfinite or out of [0, 1]")
  }
  survival_array
}

# Explainability workflow (`explainability-spec.md`, "E1 -- coefficient explanations"): a
# single-lambda bundle (the final development-scope refit, or any bundle reached only via
# --predict) has exactly one fitted
# coefficient vector. Included in every response rather than requiring a second subcommand/call, so
# a coefficient-extraction consumer (the importance/refit drivers) never pays a second R
# subprocess round trip for the same fitted state. Omitted (not merely null) when the bundle carries
# more than one lambda, since "the coefficients" is ambiguous for a whole path -- explainability
# never requests coefficients from a --fit "path" call, only from the single-lambda final-refit
# shape.
.coefficients_if_single_lambda <- function(fit, predictor_columns) {
  if (length(fit$lambda) != 1) {
    return(NULL)
  }
  as.list(stats::setNames(as.numeric(coef(fit, s = fit$lambda[1])), predictor_columns))
}

# Explainability workflow (`explainability-spec.md`'s 2026-08-11 amendment,
# `coefficient_paths.csv`): the full per-lambda coefficient trajectory for a genuine "path" fit --
# the opposite condition of
# `.coefficients_if_single_lambda` above, and the R-side half of E3's tuning-grid coefficient
# path for `glmnet_enet_core`. One list entry per lambda (in `fit$lambda` order), each a
# `{predictor: coefficient}` object, so Python-side reconstruction never has to guess column
# alignment from a bare coefficient matrix.
.coefficient_path_if_multi_lambda <- function(fit, predictor_columns) {
  if (length(fit$lambda) <= 1) {
    return(NULL)
  }
  coefficient_matrix <- as.matrix(coef(fit, s = fit$lambda))
  if (nrow(coefficient_matrix) != length(predictor_columns)) {
    fail("coefficient path row count does not match predictor_columns")
  }
  lapply(seq_along(fit$lambda), function(l) {
    as.list(stats::setNames(as.numeric(coefficient_matrix[, l]), predictor_columns))
  })
}

.write_response <- function(response_path, fit, alpha, horizons, survival_array, predictor_columns) {
  payload <- list(
    lambda = fit$lambda,
    alpha = alpha,
    horizons = horizons,
    survival = survival_array,
    coefficients = .coefficients_if_single_lambda(fit, predictor_columns),
    coefficient_path = .coefficient_path_if_multi_lambda(fit, predictor_columns),
    predictor_columns = predictor_columns,
    runtime = list(
      R = paste(R.version$major, R.version$minor, sep = "."),
      survival = as.character(utils::packageVersion("survival")),
      glmnet = as.character(utils::packageVersion("glmnet"))
    )
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA, null = "null"), response_path)
}

fit_and_predict <- function(
  train_path, val_path, destination, response_path, alpha, lambda_spec, seed, horizons_csv
) {
  suppressWarnings(suppressMessages({
    library(survival)
    library(glmnet)
  }))
  alpha <- as.numeric(alpha)
  seed <- as.integer(seed)
  set.seed(seed)
  horizons <- as.numeric(strsplit(horizons_csv, ",")[[1]])

  train_data <- utils::read.csv(train_path, check.names = FALSE)
  required_columns <- c("entry", "exit", "event")
  if (!all(required_columns %in% names(train_data))) {
    fail("Training exchange CSV is missing required entry/exit/event columns")
  }
  predictor_columns <- setdiff(names(train_data), required_columns)
  if (length(predictor_columns) == 0) {
    fail("Training exchange CSV has no predictor columns")
  }
  x_train <- as.matrix(train_data[, predictor_columns, drop = FALSE])
  y_train <- survival::Surv(train_data$entry, train_data$exit, train_data$event, type = "counting")

  fit_args <- list(
    x = x_train, y = y_train, family = "cox", alpha = alpha,
    standardize = TRUE, cox.ties = "efron"
  )
  if (identical(lambda_spec, "path")) {
    fit_args$nlambda <- 50L
    fit_args$lambda.min.ratio <- 0.01
  } else {
    # One or more explicit lambda values, comma-separated. A single value is the final
    # development-scope refit at a previously selected lambda; more than one is an inner fold
    # forced onto the parent scope's own reference lambda sequence, so every fold scores the same
    # candidate set (optimization-spec.md's tuning objective requires averaging a score across
    # folds for the *same* candidate, which only means something if every fold used the same
    # lambda values -- glmnet's own data-dependent path would otherwise differ fold to fold).
    fit_args$lambda <- as.numeric(strsplit(lambda_spec, ",")[[1]])
  }
  fit <- do.call(glmnet::glmnet, fit_args)
  if (length(fit$lambda) == 0) {
    fail("glmnet returned an empty lambda path")
  }
  # glmnet's Cox baseline hazard is not stored inside the fitted object -- `survfit.coxnet` needs
  # the original training `(x, y)` to reconstruct it even when no refitting occurs, so the bundle
  # carries them alongside the frozen, already-estimated coefficients.
  saveRDS(list(fit = fit, x_train = x_train, y_train = y_train), destination)

  x_val <- .read_predictor_matrix(val_path, predictor_columns)
  survival_array <- .predict_all_lambdas(fit, x_train, y_train, x_val, horizons)

  # Clean-process reload determinism: the saved bundle must reproduce the identical prediction for
  # the first lambda and the first assessment row.
  reloaded <- readRDS(destination)
  reload_array <- .predict_all_lambdas(
    reloaded$fit, reloaded$x_train, reloaded$y_train, x_val[1, , drop = FALSE], horizons
  )
  if (!isTRUE(all.equal(
    as.numeric(survival_array[1, 1, ]), as.numeric(reload_array[1, 1, ]), tolerance = 1e-8
  ))) {
    fail("Reloaded glmnet bundle produced a different prediction")
  }

  .write_response(response_path, fit, alpha, horizons, survival_array, predictor_columns)
}

predict_only <- function(bundle_path, val_path, response_path, horizons_csv) {
  suppressWarnings(suppressMessages({
    library(survival)
    library(glmnet)
  }))
  horizons <- as.numeric(strsplit(horizons_csv, ",")[[1]])
  bundle <- readRDS(bundle_path)
  if (is.null(bundle$fit) || is.null(bundle$x_train) || is.null(bundle$y_train)) {
    fail("Bundle is missing fit/x_train/y_train -- not a glmnet_adapter.R --fit bundle")
  }
  predictor_columns <- colnames(bundle$x_train)
  x_val <- .read_predictor_matrix(val_path, predictor_columns)
  survival_array <- .predict_all_lambdas(
    bundle$fit, bundle$x_train, bundle$y_train, x_val, horizons
  )
  .write_response(response_path, bundle$fit, NA_real_, horizons, survival_array, predictor_columns)
}

if (length(args) == 0) {
  fail("No subcommand supplied")
} else if (args[1] == "--preflight") {
  preflight()
} else if (args[1] == "--predict") {
  if (length(args) != 5) {
    fail("--predict requires bundle_rds val_csv response_json horizons_csv")
  }
  predict_only(args[2], args[3], args[4], args[5])
} else if (args[1] == "--fit") {
  if (length(args) != 9) {
    fail(paste(
      "--fit requires train_csv val_csv destination_rds response_json alpha lambda_spec seed",
      "horizons_csv"
    ))
  }
  fit_and_predict(args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9])
} else {
  fail(paste("Unknown subcommand:", args[1]))
}
