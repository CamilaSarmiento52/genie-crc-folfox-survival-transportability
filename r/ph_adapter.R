#!/usr/bin/env Rscript
# Private R adapter for the post hoc proportional-hazards assessment of the frozen Cox models.
# Invoked only via crc_folfox.revision.proportional_hazards, from a private temporary exchange.
# R evaluates Schoenfeld residuals and the Grambsch-Therneau test at the *frozen* coefficients;
# multiplicity adjustment, flags and the trigger rule stay in Python. Two subcommands:
#
#   --preflight
#       Prints one line of JSON: {"R", "survival", "jsonlite"} version strings.
#
#   --assess <train_csv> <spec_json> <response_json>
#       <train_csv> holds entry, exit, event and the design columns. <spec_json> holds
#       `coefficients` (design column -> frozen coefficient), `terms` (term name -> design
#       columns, tested jointly), `transforms`, `return_residuals` and `curve_points`.
#       Fits coxph(Surv(entry, exit, event) ~ terms, ties = "efron") with `init` at the frozen
#       coefficients and iter.max = 0, so no coefficient is re-estimated, and separately with
#       free iterations only to report how far the unpenalised optimum lies from the frozen
#       values. Runs cox.zph per transform with terms = TRUE and global = TRUE. Returns the test
#       table, smoothed residual curves (aggregate only) and, when requested for synthetic
#       fixtures, raw Schoenfeld residuals.

.script_path <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
.renv_project <- dirname(normalizePath(.script_path, mustWork = TRUE))  # r/, which holds renv.lock
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
  required <- c("survival", "jsonlite", "splines")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    fail(paste("Missing required package(s):", paste(missing, collapse = ", ")))
  }
  cat(toJSON(list(
    R = as.character(getRversion()),
    survival = as.character(packageVersion("survival")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ), auto_unbox = TRUE), "\n")
}

assess <- function(train_csv, spec_json, response_json) {
  suppressWarnings(suppressMessages({
    library(survival)
    library(splines)
  }))
  spec <- fromJSON(spec_json, simplifyVector = FALSE)
  raw <- read.csv(train_csv, check.names = FALSE)
  term_names <- names(spec$terms)
  frame <- data.frame(entry = raw$entry, exit = raw$exit, event = raw$event)
  init <- c()
  safe_names <- paste0("term", seq_along(term_names))
  for (i in seq_along(term_names)) {
    columns <- unlist(spec$terms[[i]])
    if (!all(columns %in% names(raw))) fail("Design column missing from exchange")
    frame[[safe_names[i]]] <- as.matrix(raw[, columns, drop = FALSE])
    init <- c(init, vapply(columns, function(c) as.numeric(spec$coefficients[[c]]), numeric(1)))
  }
  formula <- as.formula(paste("Surv(entry, exit, event) ~", paste(safe_names, collapse = " + ")))
  frozen <- coxph(formula, data = frame, ties = "efron", init = init,
                  control = coxph.control(iter.max = 0))
  free <- coxph(formula, data = frame, ties = "efron")

  tests <- list()
  curves <- list()
  for (transform in unlist(spec$transforms)) {
    z <- cox.zph(frozen, transform = transform, terms = TRUE, global = TRUE)
    tab <- z$table
    labels <- c(term_names, "GLOBAL")
    for (r in seq_len(nrow(tab))) {
      tests[[length(tests) + 1]] <- list(
        transform = transform, term = labels[r],
        chisq = unname(tab[r, "chisq"]), df = unname(tab[r, "df"]), p = unname(tab[r, "p"])
      )
    }
    if (transform == unlist(spec$transforms)[1]) {
      xx <- z$x
      grid <- seq(min(xx), max(xx), length.out = as.integer(spec$curve_points))
      time_at <- approx(xx, z$time, xout = grid, ties = "ordered", rule = 2)$y
      for (i in seq_along(term_names)) {
        yy <- z$y[, i]
        df_smooth <- min(4, max(1, length(unique(xx)) - 2))
        fit <- lm(yy ~ ns(xx, df = df_smooth))
        pred <- predict(fit, newdata = data.frame(xx = grid), se.fit = TRUE)
        curves[[length(curves) + 1]] <- list(
          term = term_names[i], time = time_at, fit = unname(pred$fit),
          lower = unname(pred$fit - 1.96 * pred$se.fit),
          upper = unname(pred$fit + 1.96 * pred$se.fit)
        )
      }
    }
  }
  response <- list(
    n = frozen$n, events = frozen$nevent,
    frozen_score_statistic = unname(frozen$score),
    free_coefficients = unname(coef(free)),
    frozen_coefficients = unname(init),
    tests = tests, curves = curves
  )
  if (isTRUE(spec$return_residuals)) {
    res <- residuals(frozen, type = "schoenfeld")
    response$schoenfeld <- unname(as.matrix(res))
    response$schoenfeld_times <- as.numeric(rownames(as.matrix(res)))
  }
  write(toJSON(response, auto_unbox = TRUE, digits = NA, na = "null"), response_json)
}

if (length(args) == 1 && args[1] == "--preflight") {
  preflight()
} else if (length(args) == 4 && args[1] == "--assess") {
  assess(args[2], args[3], args[4])
} else {
  fail("Usage: ph_adapter.R --preflight | --assess <train_csv> <spec_json> <response_json>")
}
