#!/usr/bin/env Rscript
# Private, schema-validated R adapter for the model-optimization flexsurv Royston-Parmar spline
# candidate (B6 `flexsurv_rp_spline`; optimization-spec.md). Invoked only via
# crc_folfox.optimize.r_adapter.FlexsurvAdapter, from a private temporary exchange -- never run
# interactively against restricted data. R fits and predicts; every validation score, knot-count
# selection, and comparison stays in Python. Three subcommands:
#
#   --preflight
#       Prints one line of JSON to stdout: {"R", "survival", "flexsurv", "jsonlite"} version
#       strings. Exit status 1 on any missing dependency.
#
#   --fit <train_csv> <val_csv> <destination_rds> <response_json> <k> <seed> <horizons_csv>
#       Fits flexsurvspline(..., scale="hazard", k=<k>) with a counting-process response on
#       <train_csv> -- <k> internal knots placed at flexsurv's own default quantiles of the
#       fitting partition's uncensored log event times (optimization-spec.md, "B6"). Predicts
#       absolute survival at every requested horizon in <horizons_csv> for every row of
#       <val_csv>. Saves the fitted flexsurvreg object, reloads it, and proves reload determinism
#       before writing the response. Unlike glmnet's Cox baseline, flexsurv's fitted parametric
#       survivor function is fully determined by its estimated coefficients, so nothing besides
#       the fitted object itself needs to be saved.
#
#   --predict <bundle_rds> <val_csv> <response_json> <horizons_csv>
#       Loads a bundle saved by --fit and predicts at every requested horizon for every row of
#       <val_csv>. No fitting occurs -- this is the evaluation-time path.

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
  required <- c("survival", "flexsurv", "jsonlite")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    fail(paste("Missing required package(s):", paste(missing, collapse = ", ")))
  }
  suppressWarnings(suppressMessages({
    library(survival)
    library(flexsurv)
  }))
  payload <- list(
    R = paste(R.version$major, R.version$minor, sep = "."),
    survival = as.character(utils::packageVersion("survival")),
    flexsurv = as.character(utils::packageVersion("flexsurv")),
    jsonlite = as.character(utils::packageVersion("jsonlite"))
  )
  cat(toJSON(payload, auto_unbox = TRUE))
}

.build_formula <- function(predictor_columns) {
  quoted <- sprintf("`%s`", predictor_columns)
  stats::as.formula(paste(
    "survival::Surv(entry, exit, event, type = \"counting\") ~", paste(quoted, collapse = " + ")
  ))
}

.predict_matrix <- function(fit, newdata, horizons) {
  summaries <- summary(fit, newdata = newdata, type = "survival", t = horizons, ci = FALSE)
  if (length(summaries) != nrow(newdata)) {
    fail("flexsurv summary() did not return one curve per assessment row")
  }
  out <- matrix(NA_real_, nrow = nrow(newdata), ncol = length(horizons))
  for (row in seq_along(summaries)) {
    values <- summaries[[row]]$est
    if (length(values) != length(horizons)) {
      fail("flexsurv summary() curve length does not match the requested horizon count")
    }
    out[row, ] <- values
  }
  out
}

fit_and_predict <- function(train_path, val_path, destination, response_path, k, seed, horizons_csv) {
  suppressWarnings(suppressMessages({
    library(survival)
    library(flexsurv)
  }))
  k <- as.integer(k)
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
  design <- .build_formula(predictor_columns)

  fit <- flexsurv::flexsurvspline(design, data = train_data, k = k, scale = "hazard")
  saveRDS(fit, destination)

  val_data <- utils::read.csv(val_path, check.names = FALSE)
  missing_columns <- setdiff(predictor_columns, names(val_data))
  if (length(missing_columns) > 0) {
    fail(paste(
      "Assessment exchange CSV is missing predictor column(s):",
      paste(missing_columns, collapse = ", ")
    ))
  }

  survival_matrix <- .predict_matrix(fit, val_data[, predictor_columns, drop = FALSE], horizons)
  if (any(!is.finite(survival_matrix)) || any(survival_matrix < 0) || any(survival_matrix > 1)) {
    fail("flexsurv survival prediction is nonfinite or out of [0, 1]")
  }

  # Clean-process reload determinism: the saved bundle must reproduce the identical prediction for
  # the first assessment row.
  reloaded <- readRDS(destination)
  reload_curve <- .predict_matrix(reloaded, val_data[1, predictor_columns, drop = FALSE], horizons)
  if (!isTRUE(all.equal(
    as.numeric(survival_matrix[1, ]), as.numeric(reload_curve[1, ]), tolerance = 1e-8
  ))) {
    fail("Reloaded flexsurv bundle produced a different prediction")
  }

  payload <- list(
    k = k,
    horizons = horizons,
    survival = survival_matrix,
    runtime = list(
      R = paste(R.version$major, R.version$minor, sep = "."),
      survival = as.character(utils::packageVersion("survival")),
      flexsurv = as.character(utils::packageVersion("flexsurv"))
    )
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA), response_path)
}

predict_only <- function(bundle_path, val_path, response_path, horizons_csv) {
  suppressWarnings(suppressMessages({
    library(survival)
    library(flexsurv)
  }))
  horizons <- as.numeric(strsplit(horizons_csv, ",")[[1]])
  fit <- readRDS(bundle_path)
  predictor_columns <- fit$covdata$covnames
  if (is.null(predictor_columns)) {
    fail("Bundle is missing covariate names -- not a flexsurv_adapter.R --fit bundle")
  }
  val_data <- utils::read.csv(val_path, check.names = FALSE)
  missing_columns <- setdiff(predictor_columns, names(val_data))
  if (length(missing_columns) > 0) {
    fail(paste(
      "Assessment exchange CSV is missing predictor column(s):",
      paste(missing_columns, collapse = ", ")
    ))
  }
  survival_matrix <- .predict_matrix(fit, val_data[, predictor_columns, drop = FALSE], horizons)
  if (any(!is.finite(survival_matrix)) || any(survival_matrix < 0) || any(survival_matrix > 1)) {
    fail("flexsurv survival prediction is nonfinite or out of [0, 1]")
  }
  payload <- list(
    k = fit$k,
    horizons = horizons,
    survival = survival_matrix,
    runtime = list(
      R = paste(R.version$major, R.version$minor, sep = "."),
      survival = as.character(utils::packageVersion("survival")),
      flexsurv = as.character(utils::packageVersion("flexsurv"))
    )
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA), response_path)
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
  if (length(args) != 8) {
    fail("--fit requires train_csv val_csv destination_rds response_json k seed horizons_csv")
  }
  fit_and_predict(args[2], args[3], args[4], args[5], args[6], args[7], args[8])
} else {
  fail(paste("Unknown subcommand:", args[1]))
}
