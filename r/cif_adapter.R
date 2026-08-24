#!/usr/bin/env Rscript
# Private, schema-validated R adapter for the forest-retry delayed-entry conditional-inference
# forest (LTRCforests::ltrccif). Invoked only via
# crc_folfox.forest.r_adapter.RConditionalForestAdapter, from a private temporary exchange --
# never run interactively against restricted data.
#
# R FITS AND PREDICTS; R NEVER SCORES. No metric, selection, or information-criterion routine
# exists here or may be added (tech-stack.md section 1, model-optimization adapter boundary).
#
# Subcommands:
#
#   --preflight
#       One line of JSON to stdout: {"R","survival","LTRCforests","partykit","jsonlite"}.
#       Exit status 1 on any missing dependency.
#
#   --fit <input_csv> <destination_rds> <response_json> <mtry> <minbucket> <seed>
#       Fits ltrccif on the frozen configuration below, saves the RDS bundle, reloads it to
#       prove clean-process reload determinism, and writes a synthetic-fixture survival
#       prediction on BOTH risk scales to <response_json>.
#
#   --predict <bundle_rds> <newdata_csv> <response_json> <times_csv>
#       Predicts absolute survival at the supplied times on BOTH risk scales.
#
# WHY TWO RISK SCALES (forest-model implementation notes, 2026-08-15).
# LTRCforests::predictProb returns
# ENTRY-CONDITIONAL survival, S(t | T > entry_i), not unconditional S(t). Verified: a row with
# entry = 0 -- for which the conditioning is vacuous -- reproduces an independent
# survival::survfit delayed-entry Kaplan-Meier to 3.3e-16 at every IBS grid point. `entry` sits
# in the response, never among the predictors, so it cannot affect terminal-node assignment;
# it changes only the conditioning. Both scales are therefore obtained from the same fitted
# forest:
#     entry_conditional -> predict at each row's own entry
#     unconditional     -> predict at entry = 0
# The project's performance artifacts carry exactly this `risk_scale` distinction, and the
# 2026-08-15 audit made reporting both mandatory wherever either is reported.
#
# NO OUTCOME LEAKAGE (verified 2026-08-15, and enforced by an automated sentinel test).
# predictProb ignores `exit` and `event` in newdata: flipping every event indicator or tripling
# every exit time leaves predictions bit-identical, while shifting entry alone moves them. Only
# `entry` and the covariates are used.

# Rscript does not auto-source a project .Rprofile unless invoked with that project as the
# working directory. Activate this script's own project-scoped renv library explicitly, based on
# its own file path, so the pinned package versions are found regardless of the caller's working
# directory. (Regression discovered during primary model training: without this the global R
# library was used silently.)
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

# ---- Frozen configuration (forest-retry-spec.md section 4) --------------------------------
# Only `mtry` and the terminal node size (`minbucket`) are tuned. Everything else is fixed
# before any real fit. ntree is 100 -- ltrccif's OWN package default -- per the dated 2026-08-15
# amendment in forest-retry-spec.md; 500 trees were locked for the different ltrcrrf algorithm
# and projected to 7-13 days of serial compute. Fewer trees means higher ensemble variance, so
# this concession costs the candidate rather than helping it.
# bootstrap/samptype/sampfrac/stepFactor match the original relative-risk forest configuration,
# which also coincides with this package's own defaults for sampfrac and samptype. The remaining
# ctree_control arguments are the package's documented defaults, including mincriterion = 0
# (no pre-pruning, standard for a forest ensemble).
CIF_NTREE <- 100L
CIF_BOOTSTRAP <- "by.root"
CIF_SAMPTYPE <- "swor"
CIF_SAMPFRAC <- 0.632
CIF_STEPFACTOR <- 2

cif_control <- function(n, minbucket) {
  partykit::ctree_control(
    teststat = "quad",
    testtype = "Univ",
    minsplit = max(ceiling(sqrt(n)), 20L),
    minbucket = as.integer(minbucket),
    minprob = 0.01,
    mincriterion = 0,
    saveinfo = FALSE
  )
}

load_packages <- function() {
  suppressWarnings(suppressMessages({
    library(survival)
    library(partykit)
    library(LTRCforests)
  }))
}

preflight <- function() {
  required <- c("survival", "partykit", "LTRCforests", "jsonlite")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    fail(paste("Missing required package(s):", paste(missing, collapse = ", ")))
  }
  load_packages()
  payload <- list(
    R = paste(R.version$major, R.version$minor, sep = "."),
    survival = as.character(utils::packageVersion("survival")),
    LTRCforests = as.character(utils::packageVersion("LTRCforests")),
    partykit = as.character(utils::packageVersion("partykit")),
    jsonlite = as.character(utils::packageVersion("jsonlite"))
  )
  cat(toJSON(payload, auto_unbox = TRUE))
}

read_exchange <- function(path) {
  data <- utils::read.csv(path, check.names = FALSE)
  required_columns <- c("entry", "exit", "event")
  if (!all(required_columns %in% names(data))) {
    fail("Exchange CSV is missing required entry/exit/event columns")
  }
  predictors <- setdiff(names(data), required_columns)
  if (length(predictors) == 0) {
    fail("Exchange CSV has no predictor columns")
  }
  list(data = data, predictors = predictors)
}

# Both risk scales from one fitted forest. `unconditional` sets entry to 0; `entry_conditional`
# keeps each row's own entry. `exit`/`event` are carried only because predictProb wants a
# well-formed frame -- it demonstrably ignores them.
predict_both_scales <- function(forest, newdata, times) {
  conditional <- newdata
  unconditional <- newdata
  unconditional$entry <- 0

  as_matrix <- function(nd) {
    prediction <- LTRCforests::predictProb(
      object = forest, newdata = nd, OOB = FALSE, time.eval = times
    )
    probs <- prediction$survival.probs
    # predictProb returns times x observations; transpose to observations x times.
    matrix(as.numeric(probs), nrow = length(times), ncol = nrow(nd))
  }
  list(
    entry_conditional = t(as_matrix(conditional)),
    unconditional = t(as_matrix(unconditional))
  )
}

fit_and_predict <- function(input_path, destination, response_path, mtry, minbucket, seed,
                            bootstrap = CIF_BOOTSTRAP) {
  load_packages()
  mtry <- as.integer(mtry)
  minbucket <- as.integer(minbucket)
  seed <- as.integer(seed)
  set.seed(seed)
  # `bootstrap` is overridable ONLY by the admission gate's hand-calculated-fixture check
  # (gate 3), which needs every tree to see the full sample so a deliberately unsplittable
  # forest reduces to the delayed-entry Kaplan-Meier of all training rows. Real fits never
  # reach this argument: the Python `fit()` method does not expose it.
  if (!bootstrap %in% c(CIF_BOOTSTRAP, "none")) {
    fail(paste("bootstrap must be", CIF_BOOTSTRAP, "or (admission gate only) none"))
  }

  parsed <- read_exchange(input_path)
  data <- parsed$data
  predictors <- parsed$predictors
  design <- stats::as.formula(
    paste("Surv(entry, exit, event) ~", paste(predictors, collapse = " + "))
  )

  forest <- LTRCforests::ltrccif(
    formula = design,
    data = data,
    ntree = CIF_NTREE,
    mtry = mtry,
    bootstrap = bootstrap,
    samptype = CIF_SAMPTYPE,
    sampfrac = CIF_SAMPFRAC,
    stepFactor = CIF_STEPFACTOR,
    control = cif_control(nrow(data), minbucket),
    trace = FALSE
  )

  saveRDS(forest, destination)

  times <- c(365, 730)
  newdata <- data[1, , drop = FALSE]
  scales <- predict_both_scales(forest, newdata, times)

  # Clean-process reload: the saved RDS must reload and reproduce identical predictions on both
  # scales (the admission gate's "deterministic state reproduction" requirement).
  reloaded <- readRDS(destination)
  reload_scales <- predict_both_scales(reloaded, newdata, times)
  for (scale in names(scales)) {
    if (!isTRUE(all.equal(scales[[scale]], reload_scales[[scale]]))) {
      fail(paste("Reloaded forest produced a different prediction on scale:", scale))
    }
  }

  payload <- list(
    trees = CIF_NTREE,
    survival_unconditional = scales$unconditional,
    survival_entry_conditional = scales$entry_conditional,
    runtime = list(
      R = paste(R.version$major, R.version$minor, sep = "."),
      survival = as.character(utils::packageVersion("survival")),
      LTRCforests = as.character(utils::packageVersion("LTRCforests")),
      partykit = as.character(utils::packageVersion("partykit"))
    )
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA), response_path)
}

predict_bundle <- function(bundle_path, newdata_path, response_path, times_path) {
  load_packages()
  forest <- readRDS(bundle_path)
  parsed <- read_exchange(newdata_path)
  times <- as.numeric(utils::read.csv(times_path, check.names = FALSE)[["time"]])
  if (length(times) == 0 || any(!is.finite(times))) {
    fail("Prediction time grid is empty or non-finite")
  }
  scales <- predict_both_scales(forest, parsed$data, times)
  payload <- list(
    times = times,
    survival_unconditional = scales$unconditional,
    survival_entry_conditional = scales$entry_conditional
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA), response_path)
}

if (length(args) == 0) {
  fail("No subcommand supplied")
} else if (args[1] == "--preflight") {
  preflight()
} else if (args[1] == "--fit") {
  if (!length(args) %in% c(7L, 8L)) {
    fail("--fit requires input_csv destination response_json mtry minbucket seed [bootstrap]")
  }
  bootstrap <- if (length(args) == 8L) args[8] else CIF_BOOTSTRAP
  fit_and_predict(args[2], args[3], args[4], args[5], args[6], args[7], bootstrap)
} else if (args[1] == "--predict") {
  if (length(args) != 5) {
    fail("--predict requires bundle_rds newdata_csv response_json times_csv")
  }
  predict_bundle(args[2], args[3], args[4], args[5])
} else {
  fail(paste("Unknown subcommand:", args[1]))
}
