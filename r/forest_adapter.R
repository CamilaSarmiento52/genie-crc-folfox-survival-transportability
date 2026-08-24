#!/usr/bin/env Rscript
# Private, schema-validated R adapter for the primary-model delayed-entry relative-risk forest
# admission gate. Invoked only via crc_folfox.model_training.r_adapter.RForestAdapter, from a
# private temporary exchange -- never run interactively against restricted data. Two
# subcommands, matching the Python adapter's subprocess contract exactly:
#
#   --preflight
#       Prints one line of JSON to stdout: {"R", "survival", "LTRCforests", "jsonlite"}
#       version strings. Exit status 1 on any missing dependency.
#
#   --fit <input_csv> <destination_rds> <response_json> <mtry> <nodesize> <seed>
#       Fits LTRCforests::ltrcrrf on the locked configuration (model-training-spec.md,
#       "Conditional candidate 5"), saves the RDS bundle, reloads it to prove clean-process
#       reload determinism, and writes a synthetic-fixture survival prediction at 365/730 days
#       to <response_json> for the admission gate's own inspection.

# Rscript does not auto-source a project .Rprofile unless invoked with that project as the
# working directory, and the Python caller (crc_folfox.model_training.r_adapter.RForestAdapter)
# invokes this script by absolute path without setting `cwd`. Activate this script's own
# project-scoped renv library explicitly, based on its own file path, so the pinned
# survival/LTRCforests/jsonlite versions are found regardless of the caller's working directory.
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
  required <- c("survival", "LTRCforests", "jsonlite")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    fail(paste("Missing required package(s):", paste(missing, collapse = ", ")))
  }
  suppressWarnings(suppressMessages({
    library(survival)
    library(LTRCforests)
  }))
  payload <- list(
    R = paste(R.version$major, R.version$minor, sep = "."),
    survival = as.character(utils::packageVersion("survival")),
    LTRCforests = as.character(utils::packageVersion("LTRCforests")),
    jsonlite = as.character(utils::packageVersion("jsonlite"))
  )
  cat(toJSON(payload, auto_unbox = TRUE))
}

fit_and_predict <- function(input_path, destination, response_path, mtry, nodesize, seed) {
  suppressWarnings(suppressMessages({
    library(survival)
    library(LTRCforests)
  }))
  mtry <- as.integer(mtry)
  nodesize <- as.integer(nodesize)
  seed <- as.integer(seed)
  set.seed(seed)

  data <- utils::read.csv(input_path, check.names = FALSE)
  required_columns <- c("entry", "exit", "event")
  if (!all(required_columns %in% names(data))) {
    fail("Exchange CSV is missing required entry/exit/event columns")
  }
  predictors <- setdiff(names(data), required_columns)
  if (length(predictors) == 0) {
    fail("Exchange CSV has no predictor columns")
  }
  design <- stats::as.formula(
    paste("Surv(entry, exit, event) ~", paste(predictors, collapse = " + "))
  )

  forest <- LTRCforests::ltrcrrf(
    formula = design,
    data = data,
    ntree = 500L,
    mtry = mtry,
    nodesize = nodesize,
    bootstrap = "by.root",
    samptype = "swor",
    sampfrac = 0.632,
    stepFactor = 2,
    nodedepth = NULL,
    nsplit = 10L,
    trace = FALSE
  )

  saveRDS(forest, destination)

  newdata <- data[1, predictors, drop = FALSE]
  prediction <- LTRCforests::predictProb(
    object = forest, newdata = newdata, OOB = FALSE, time.eval = c(365, 730)
  )
  survival_matrix <- matrix(as.numeric(prediction$survival.probs), nrow = 1, ncol = 2)

  # Clean-process reload: the saved RDS must reload and reproduce the identical synthetic
  # prediction (the admission gate's own "deterministic state reproduction" requirement).
  reloaded <- readRDS(destination)
  reload_prediction <- LTRCforests::predictProb(
    object = reloaded, newdata = newdata, OOB = FALSE, time.eval = c(365, 730)
  )
  reload_matrix <- matrix(as.numeric(reload_prediction$survival.probs), nrow = 1, ncol = 2)
  if (!isTRUE(all.equal(survival_matrix, reload_matrix))) {
    fail("Reloaded forest produced a different synthetic prediction")
  }

  payload <- list(
    trees = forest$ntree,
    survival = survival_matrix,
    runtime = list(
      R = paste(R.version$major, R.version$minor, sep = "."),
      survival = as.character(utils::packageVersion("survival")),
      LTRCforests = as.character(utils::packageVersion("LTRCforests"))
    )
  )
  writeLines(toJSON(payload, auto_unbox = TRUE), response_path)
}

if (length(args) == 0) {
  fail("No subcommand supplied")
} else if (args[1] == "--preflight") {
  preflight()
} else if (args[1] == "--fit") {
  if (length(args) != 7) {
    fail("--fit requires input_csv destination response_json mtry nodesize seed")
  }
  fit_and_predict(args[2], args[3], args[4], args[5], args[6], args[7])
} else {
  fail(paste("Unknown subcommand:", args[1]))
}
