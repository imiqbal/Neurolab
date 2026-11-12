#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

# =============================================================================
#  NeuroLab v8.0  —  Interactive Neural Network Studio (R / Shiny)
#  Developer: M. Iqbal Jeelani, Arif Bashir— SKUAST-Kashmir
#
#  v8 highlights
#  -------------
#  • Robust demo loader (fixes the Pima error: "invalid 'type' (list)" /
#    "'data' must be 2-dimensional" when mlbench's data() returned nothing).
#    Pima falls back to MASS::Pima.tr + Pima.te, which ship with every R.
#  • AI Insights: a built-in interpretation engine writes a statistician-grade
#    narrative (performance grading vs baseline, over-fitting check, key drivers
#    with direction, error analysis, recommendations). Optional Claude (Anthropic
#    API) narrative on top when you supply an API key.
#  • One-click Word (.docx) report with tables, publication-quality figures and
#    the full AI interpretation (officer + flextable).
#  • Five colour themes, animated training overlay, count-up KPIs, partial
#    dependence (PDP / ICE) plots, Sankey weight-flow view, 3-D / scatter-matrix
#    / violin / parallel-coordinate explorers.
#  • Plots use a colour-blind-validated categorical palette in fixed order,
#    single-hue sequential scales and a blue↔red diverging scale for weights.
#
#  Engine (from v7): auto-recovery from non-convergence, safe column names,
#  one-hot encoding + imputation, scaled regression targets, stratified split,
#  permutation & Olden importance, k-fold tuning, batch scoring.
#
#  Packages: shiny bslib neuralnet plotly DT ggplot2 mlbench MASS
#            officer flextable (Word report)   httr2 (optional, Claude AI)
#            shinycssloaders (optional)
# =============================================================================

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(neuralnet)
  library(plotly)
  library(DT)
  library(ggplot2)
})

HAS_SPINNER   <- requireNamespace("shinycssloaders", quietly = TRUE)
HAS_OFFICER   <- requireNamespace("officer", quietly = TRUE) && requireNamespace("flextable", quietly = TRUE)
HAS_HTTR2     <- requireNamespace("httr2", quietly = TRUE)
HAS_RAGG      <- requireNamespace("ragg", quietly = TRUE)

spin <- function(x) {
  if (HAS_SPINNER) shinycssloaders::withSpinner(x, type = 8, color = "#7C3AED") else x
}

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# ─────────────────────────────────────────────────────────────────────────────
#  ENGINE  ── data preparation
# ─────────────────────────────────────────────────────────────────────────────

to_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

# Fraction of non-missing entries that parse as numbers
num_share <- function(x) {
  x <- x[!is.na(x)]
  if (is.character(x)) x <- trimws(x[trimws(x) != ""])
  if (!length(x)) return(0)
  mean(!is.na(to_num(x)))
}

# Read CSV / TSV / semicolon files with automatic separator detection
read_any <- function(path) {
  first <- readLines(path, n = 5, warn = FALSE)
  first <- first[nzchar(first)]
  if (!length(first)) stop("The file is empty.")
  cnt <- sapply(c(",", ";", "\t", "|"), function(s)
    mean(lengths(regmatches(first, gregexpr(s, first, fixed = TRUE)))))
  sep <- names(cnt)[which.max(cnt)]
  if (max(cnt) == 0) sep <- ","
  dec <- if (sep == ";") "," else "."
  d <- utils::read.table(path, header = TRUE, sep = sep, dec = dec, quote = "\"'",
                         check.names = FALSE, stringsAsFactors = FALSE,
                         na.strings = c("", "NA", "N/A", "NaN", "na", "?", "-", "null", "NULL"),
                         comment.char = "", strip.white = TRUE, fill = TRUE)
  names(d) <- trimws(names(d))
  names(d)[names(d) == ""] <- paste0("V", which(names(d) == ""))
  names(d) <- make.unique(names(d))
  d <- d[, colSums(!is.na(d)) > 0, drop = FALSE]          # drop empty columns
  d <- d[rowSums(!is.na(d)) > 0, , drop = FALSE]          # drop empty rows
  if (ncol(d) < 2) stop("Need at least 2 columns (predictors + target). Check the separator.")
  d
}

# Decide classification vs regression for a target vector
detect_type <- function(y) {
  y <- y[!is.na(y)]
  if (is.logical(y) || is.factor(y)) return("classification")
  if (is.character(y) && num_share(y) < 0.95) return("classification")
  yn <- to_num(y); yn <- yn[!is.na(yn)]
  u <- length(unique(yn))
  if (u <= 10 && all(yn == round(yn))) return("classification")
  "regression"
}

# Learn the preprocessing "recipe" from TRAINING predictors only
fit_recipe <- function(X, scaling = "minmax", max_levels = 20) {
  num <- list(); cat <- list(); dropped <- character()
  for (col in names(X)) {
    v <- X[[col]]
    if (is.logical(v)) v <- as.numeric(v)
    is_numeric <- is.numeric(v) || (!is.factor(v) && num_share(v) >= 0.95)
    if (is_numeric) {
      vn <- to_num(v)
      if (all(is.na(vn)) || length(unique(na.omit(vn))) < 2) {
        dropped <- c(dropped, paste0(col, " (constant)")); next
      }
      med <- stats::median(vn, na.rm = TRUE)
      vn[is.na(vn)] <- med
      p <- switch(scaling,
                  minmax = c(min(vn), max(vn) - min(vn)),
                  zscore = c(mean(vn), stats::sd(vn)),
                  c(0, 1))
      if (!is.finite(p[2]) || p[2] == 0) p[2] <- 1
      num[[col]] <- list(median = med, center = p[1], scale = p[2],
                         min = min(vn), max = max(vn))
    } else {
      vc <- trimws(as.character(v)); vc[vc == ""] <- NA
      lev <- sort(unique(na.omit(vc)))
      if (length(lev) < 2) { dropped <- c(dropped, paste0(col, " (single level)")); next }
      if (length(lev) > max_levels) {
        dropped <- c(dropped, paste0(col, " (", length(lev), " levels — ID-like)")); next
      }
      mode <- names(sort(table(vc), decreasing = TRUE))[1]
      dums <- if (length(lev) == 2) lev[2] else lev       # binary → one dummy
      cat[[col]] <- list(levels = lev, mode = mode, dummies = dums)
    }
  }
  if (!length(num) && !length(cat)) stop("No usable predictor columns were found.")
  
  disp <- c(names(num),
            unlist(lapply(names(cat), function(cl) paste0(cl, ": ", cat[[cl]]$dummies)), use.names = FALSE))
  group <- c(names(num),
             unlist(lapply(names(cat), function(cl) rep(cl, length(cat[[cl]]$dummies))), use.names = FALSE))
  list(num = num, cat = cat, dropped = dropped, scaling = scaling,
       features = disp, groups = group, internal = paste0("X", seq_along(disp)),
       vars = c(names(num), names(cat)))
}

# Apply recipe → plain numeric matrix with internal column names
bake <- function(rec, df) {
  miss <- setdiff(rec$vars, names(df))
  if (length(miss)) stop("Missing column(s): ", paste(miss, collapse = ", "))
  n <- nrow(df)
  out <- vector("list", length(rec$features)); k <- 0
  for (col in names(rec$num)) {
    r <- rec$num[[col]]
    v <- to_num(df[[col]]); v[is.na(v)] <- r$median
    k <- k + 1; out[[k]] <- (v - r$center) / r$scale
  }
  for (col in names(rec$cat)) {
    r <- rec$cat[[col]]
    v <- trimws(as.character(df[[col]])); v[is.na(v) | v == ""] <- r$mode
    for (lv in r$dummies) { k <- k + 1; out[[k]] <- as.numeric(v == lv) }
  }
  M <- matrix(unlist(out), nrow = n, ncol = length(out))
  colnames(M) <- rec$internal
  M
}

stratified_split <- function(y, frac, seed) {
  set.seed(seed)
  n <- length(y)
  if (is.factor(y)) {
    idx <- unlist(lapply(split(seq_len(n), y), function(ix) {
      if (length(ix) <= 1) return(ix)
      k <- max(1, min(length(ix) - 1, round(frac * length(ix))))
      ix[sample.int(length(ix), k)]
    }), use.names = FALSE)
  } else {
    k <- max(5, min(n - 3, floor(frac * n)))
    idx <- sample.int(n, k)
  }
  sort(idx)
}

# Full preparation: clean target, split, fit recipe on train, encode
nl_prepare <- function(df, target, type = "auto", exclude = character(),
                       train_frac = 0.75, scaling = "minmax", seed = 2024) {
  if (!target %in% names(df)) stop("Target column '", target, "' not found.")
  y_raw <- df[[target]]
  type <- if (type == "auto") detect_type(y_raw) else type
  msgs <- character()
  
  if (type == "regression") {
    y <- to_num(y_raw)
    if (mean(is.na(y[!is.na(y_raw)])) > 0.05)
      stop("Target '", target, "' is not numeric — choose Classification instead.")
  } else {
    yc <- trimws(as.character(y_raw)); yc[yc == ""] <- NA
    y <- factor(yc)
  }
  keep <- !is.na(y)
  if (sum(!keep)) msgs <- c(msgs, paste(sum(!keep), "rows with missing target removed"))
  df <- df[keep, , drop = FALSE]; y <- y[keep]
  
  if (type == "classification") {
    tab <- table(y)
    rare <- names(tab)[tab < 2]
    if (length(rare)) {
      msgs <- c(msgs, paste0("Classes with <2 rows removed: ", paste(rare, collapse = ", ")))
      k <- !(y %in% rare); df <- df[k, , drop = FALSE]; y <- droplevels(y[k])
    }
    if (nlevels(y) < 2) stop("Target needs at least 2 classes.")
    if (nlevels(y) > 30) stop("Target has ", nlevels(y), " classes — is it really categorical? Choose Regression.")
  }
  if (length(y) < 15) stop("Too few usable rows (", length(y), "). Need at least 15.")
  
  preds <- setdiff(names(df), c(target, exclude))
  if (!length(preds)) stop("No predictor columns selected.")
  X <- df[, preds, drop = FALSE]
  
  idx <- stratified_split(y, train_frac, seed)
  rec <- fit_recipe(X[idx, , drop = FALSE], scaling)
  if (length(rec$dropped)) msgs <- c(msgs, paste("Dropped:", paste(rec$dropped, collapse = "; ")))
  Xtr <- bake(rec, X[idx, , drop = FALSE]); Xte <- bake(rec, X[-idx, , drop = FALSE])
  
  if (type == "regression") {
    ymin <- min(y[idx]); yr <- max(y[idx]) - ymin; if (yr == 0) yr <- 1
    rec$y <- list(min = ymin, range = yr)
    Ytr <- matrix((y[idx] - ymin) / yr, ncol = 1, dimnames = list(NULL, "Y1"))
    lev <- NULL
  } else {
    lev <- levels(y)
    Ytr <- if (length(lev) == 2) {
      matrix(as.numeric(y[idx] == lev[2]), ncol = 1, dimnames = list(NULL, "Y1"))
    } else {
      M <- sapply(lev, function(l) as.numeric(y[idx] == l))
      colnames(M) <- paste0("Y", seq_along(lev)); M
    }
  }
  rec$type <- type; rec$lev <- lev; rec$target <- target
  
  list(rec = rec, Xtr = Xtr, Xte = Xte, Ytr = Ytr,
       ytr = y[idx], yte = y[-idx], idx = idx,
       raw_test = df[-idx, , drop = FALSE], raw_train = df[idx, , drop = FALSE], raw_all = df,
       type = type, lev = lev, msgs = msgs)
}

# ─────────────────────────────────────────────────────────────────────────────
#  ENGINE  ── training / prediction
# ─────────────────────────────────────────────────────────────────────────────

ACT_FUNS <- list(
  logistic = "logistic",
  tanh     = "tanh",
  softplus = function(x) log(1 + exp(x))     # smooth ReLU; differentiable for neuralnet
)
act_eval <- function(name) switch(name,
                                  logistic = function(x) 1 / (1 + exp(-x)),
                                  tanh     = tanh,
                                  softplus = function(x) pmax(x, 0) + log1p(exp(-abs(x))),   # numerically stable
                                  function(x) 1 / (1 + exp(-x)))

ACT_LABELS <- c(logistic = "Logistic (sigmoid)", tanh = "Tanh", softplus = "Softplus (smooth ReLU)")

nl_fit_once <- function(Xtr, Ytr, type, hidden, act, algorithm, lr, threshold,
                        stepmax, reps, err, seed) {
  dat <- data.frame(Xtr, Ytr, check.names = FALSE)
  fml <- stats::as.formula(paste(paste(colnames(Ytr), collapse = " + "), "~",
                                 paste(colnames(Xtr), collapse = " + ")))
  lin <- type == "regression"
  if (err == "auto") err <- if (!lin && act == "logistic") "ce" else "sse"
  if (err == "ce" && (lin || act != "logistic")) err <- "sse"   # ce needs (0,1) outputs
  args <- list(formula = fml, data = dat, hidden = hidden, threshold = threshold,
               stepmax = stepmax, rep = reps, algorithm = algorithm,
               err.fct = err, act.fct = ACT_FUNS[[act]], linear.output = lin)
  if (algorithm == "backprop") args$learningrate <- lr
  set.seed(seed)
  warns <- character()
  m <- withCallingHandlers(do.call(neuralnet::neuralnet, args),
                           warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  list(model = m, err = err, warns = warns)
}

# neuralnet's threshold applies to the gradient of the SUMMED error, so it must
# grow with the number of rows and outputs to stay equally strict.
auto_threshold <- function(n, n_out = 1, type = "classification") {
  k <- if (type == "regression") 0.01 else 0.03
  signif(min(0.25, max(0.01, k * n_out^0.5 * n / 100)), 1)
}

# Train with automatic threshold relaxation when the network does not converge
nl_train <- function(prep, hidden, act = "logistic", algorithm = "rprop+", lr = 0.01,
                     threshold = 0.01, stepmax = 1e5, reps = 1, err = "auto",
                     seed = 2024, auto_relax = TRUE, max_tries = 4) {
  hidden <- as.integer(hidden[hidden > 0])
  if (!length(hidden)) hidden <- 4L
  if (identical(threshold, "auto") || is.na(suppressWarnings(as.numeric(threshold))))
    threshold <- auto_threshold(nrow(prep$Xtr), ncol(prep$Ytr), prep$type)
  threshold <- as.numeric(threshold)
  t0 <- Sys.time(); thr <- threshold; notes <- character(); fit <- NULL
  tries <- if (auto_relax) max_tries else 1
  for (i in seq_len(tries)) {
    fit <- tryCatch(nl_fit_once(prep$Xtr, prep$Ytr, prep$type, hidden, act, algorithm,
                                lr, thr, stepmax, reps, err, seed),
                    error = function(e) list(model = NULL, error = conditionMessage(e)))
    ok <- !is.null(fit$model) && !is.null(fit$model$weights) && length(fit$model$weights) > 0
    if (ok) {
      bad <- any(!is.finite(unlist(fit$model$weights)))
      if (!bad) break
      fit$error <- "weights became NaN/Inf (training diverged)"
    }
    msg <- fit$error %||% "did not converge within Max Steps"
    diverged <- !is.null(fit$error) || isTRUE(fit$diverged)
    if (i < tries) {
      if (algorithm == "backprop" && diverged) {
        notes <- c(notes, sprintf("Try %d (learning rate %g): training diverged → lowering learning rate.", i, lr))
        lr <- lr / 10
      } else {
        notes <- c(notes, sprintf("Try %d (threshold %g): %s → relaxing threshold.", i, thr, msg))
        thr <- signif(thr * 3, 2)
      }
    } else {
      hint <- if (algorithm == "backprop") " Lower the learning rate or switch to rprop+." else
        " Increase Max Steps, raise the threshold, or use fewer neurons."
      stop("Training failed: ", msg, ".", hint)
    }
  }
  m <- fit$model
  errs <- m$result.matrix["error", ]
  best <- which.min(errs)
  if (length(fit$warns)) notes <- c(notes, unique(fit$warns))
  list(model = m, best = best, act = act, hidden = hidden, algorithm = algorithm,
       lr = lr, threshold = thr, threshold_req = threshold, stepmax = stepmax,
       reps_ok = ncol(m$result.matrix), reps_req = reps, err = fit$err,
       steps = unname(m$result.matrix["steps", best]), train_error = unname(errs[best]),
       secs = as.numeric(difftime(Sys.time(), t0, units = "secs")), notes = notes)
}

# Forward pass straight through the weight matrices (version-proof)
nn_forward <- function(tr, X, rep = tr$best) {
  W <- tr$model$weights[[rep]]
  f <- act_eval(tr$act)
  out <- X; L <- length(W)
  for (i in seq_len(L)) {
    out <- cbind(1, out) %*% W[[i]]
    if (i < L || !isTRUE(tr$model$linear.output)) out <- f(out)
  }
  out
}

# Hidden-layer activations (for visualisation)
nn_layer_outputs <- function(tr, X, rep = tr$best) {
  W <- tr$model$weights[[rep]]; f <- act_eval(tr$act)
  res <- list(); out <- X
  for (i in seq_along(W)) {
    out <- cbind(1, out) %*% W[[i]]
    if (i < length(W) || !isTRUE(tr$model$linear.output)) out <- f(out)
    res[[i]] <- out
  }
  res
}

decode_output <- function(raw, rec) {
  if (rec$type == "regression") {
    # Bound to training target range ± 25 % — guards against wild extrapolation
    # of unbounded activations (tanh / softplus) far outside the training data.
    s <- pmin(pmax(as.numeric(raw[, 1]), -0.25), 1.25)
    return(list(pred = s * rec$y$range + rec$y$min, prob = NULL))
  }
  lev <- rec$lev
  if (ncol(raw) == 1) {
    p2 <- pmin(pmax(raw[, 1], 0), 1)
    prob <- cbind(1 - p2, p2)
  } else {
    o <- pmax(raw, 1e-6); prob <- o / rowSums(o)
  }
  colnames(prob) <- lev
  list(pred = factor(lev[max.col(prob, ties.method = "first")], levels = lev), prob = prob)
}

nl_predict_matrix <- function(fit, X) decode_output(nn_forward(fit$train, X), fit$rec)
nl_predict_df <- function(fit, df) nl_predict_matrix(fit, bake(fit$rec, df))

# ─────────────────────────────────────────────────────────────────────────────
#  ENGINE  ── metrics & interpretation
# ─────────────────────────────────────────────────────────────────────────────

reg_metrics <- function(a, p) {
  e <- p - a
  sst <- sum((a - mean(a))^2)
  nz <- abs(a) > 1e-9
  c(RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)),
    R2 = if (sst > 0) 1 - sum(e^2) / sst else NA,
    r = if (stats::sd(a) > 0 && stats::sd(p) > 0) stats::cor(a, p) else NA,
    MAPE = if (any(nz)) 100 * mean(abs(e[nz] / a[nz])) else NA)
}

auc_binary <- function(y01, score) {
  n1 <- sum(y01 == 1); n0 <- sum(y01 == 0)
  if (!n1 || !n0) return(NA)
  r <- rank(score)
  (sum(r[y01 == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

cls_metrics <- function(a, p, prob = NULL) {
  lev <- levels(a); p <- factor(as.character(p), levels = lev)
  cm <- table(Actual = a, Predicted = p)
  n <- sum(cm); po <- sum(diag(cm)) / n
  pe <- sum(rowSums(cm) * colSums(cm)) / n^2
  prec <- diag(cm) / pmax(colSums(cm), 1)
  rec  <- diag(cm) / pmax(rowSums(cm), 1)
  f1   <- ifelse(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
  auc <- if (length(lev) == 2 && !is.null(prob)) auc_binary(as.numeric(a == lev[2]), prob[, 2]) else NA
  list(overall = c(Accuracy = po, Kappa = if (pe < 1) (po - pe) / (1 - pe) else NA,
                   BalAcc = mean(rec), MacroF1 = mean(f1), AUC = auc),
       per_class = data.frame(Class = lev, Precision = as.numeric(prec), Recall = as.numeric(rec),
                              F1 = as.numeric(f1), Support = as.integer(rowSums(cm))),
       cm = cm)
}

score_fn <- function(fit, X, y) {
  out <- nl_predict_matrix(fit, X)
  if (fit$rec$type == "regression") sqrt(mean((out$pred - y)^2)) else mean(out$pred == y)
}

# Permutation importance on the test set, grouped by original variable
perm_importance <- function(fit, X, y, n_rep = 5, seed = 1) {
  set.seed(seed)
  base <- score_fn(fit, X, y)
  grp <- fit$rec$groups
  res <- sapply(unique(grp), function(g) {
    cols <- which(grp == g)
    vals <- replicate(n_rep, {
      Xp <- X; Xp[, cols] <- X[sample.int(nrow(X)), cols, drop = FALSE]
      score_fn(fit, Xp, y)
    })
    if (fit$rec$type == "regression") mean(vals) - base else base - mean(vals)
  })
  data.frame(Feature = unique(grp), Importance = as.numeric(res), stringsAsFactors = FALSE)
}

# Olden's connection-weight importance (works for any depth)
olden_importance <- function(fit) {
  W <- fit$train$model$weights[[fit$train$best]]
  M <- W[[1]][-1, , drop = FALSE]
  if (length(W) > 1) for (i in 2:length(W)) M <- M %*% W[[i]][-1, , drop = FALSE]
  val <- if (ncol(M) == 1) M[, 1] else rowSums(abs(M))
  data.frame(Feature = fit$rec$features, Importance = as.numeric(val), stringsAsFactors = FALSE)
}

# One complete evaluation bundle
nl_evaluate <- function(prep, tr) {
  fit <- list(rec = prep$rec, train = tr)
  pte <- nl_predict_matrix(fit, prep$Xte)
  ptr <- nl_predict_matrix(fit, prep$Xtr)
  if (prep$type == "regression") {
    mt <- reg_metrics(prep$yte, pte$pred); mtr <- reg_metrics(prep$ytr, ptr$pred)
  } else {
    mt <- cls_metrics(prep$yte, pte$pred, pte$prob); mtr <- cls_metrics(prep$ytr, ptr$pred, ptr$prob)
  }
  fit$test <- list(actual = prep$yte, pred = pte$pred, prob = pte$prob, metrics = mt)
  fit$trainset <- list(actual = prep$ytr, pred = ptr$pred, prob = ptr$prob, metrics = mtr)
  fit$perm <- tryCatch(perm_importance(fit, prep$Xte, prep$yte), error = function(e) NULL)
  fit$olden <- tryCatch(olden_importance(fit), error = function(e) NULL)
  fit$raw_test <- prep$raw_test
  fit$raw_train <- prep$raw_train
  fit$n_rows_used <- nrow(prep$raw_all)
  fit$msgs <- prep$msgs
  fit$n <- c(train = nrow(prep$Xtr), test = nrow(prep$Xte))
  fit
}

# k-fold / holdout evaluation for tuning
cv_score <- function(df, target, type, exclude, scaling, hidden, act, algorithm, lr,
                     threshold, stepmax, folds, seed) {
  if (folds <= 1) {
    prep <- nl_prepare(df, target, type, exclude, 0.75, scaling, seed)
    tr <- nl_train(prep, hidden, act, algorithm, lr, threshold, stepmax, 1, "auto", seed, TRUE, 3)
    fit <- list(rec = prep$rec, train = tr)
    return(c(score = unname(score_fn(fit, prep$Xte, prep$yte)), steps = unname(tr$steps), secs = unname(tr$secs)))
  }
  base <- nl_prepare(df, target, type, exclude, 0.75, scaling, seed)
  y <- c(as.character(base$ytr), as.character(base$yte))[order(c(base$idx, setdiff(seq_len(nrow(base$raw_all)), base$idx)))]
  set.seed(seed)
  fold_id <- if (base$type == "classification") {
    ave(seq_along(y), y, FUN = function(ix) sample(rep_len(seq_len(folds), length(ix))))
  } else sample(rep_len(seq_len(folds), length(y)))
  sc <- st <- se <- numeric()
  dfa <- base$raw_all
  for (k in seq_len(folds)) {
    tr_idx <- which(fold_id != k)
    rec <- fit_recipe(dfa[tr_idx, setdiff(names(dfa), c(target, exclude)), drop = FALSE], scaling)
    rec$type <- base$type; rec$lev <- base$lev
    Xtr <- bake(rec, dfa[tr_idx, , drop = FALSE]); Xte <- bake(rec, dfa[-tr_idx, , drop = FALSE])
    if (base$type == "regression") {
      yy <- to_num(dfa[[target]]); ymin <- min(yy[tr_idx]); yr <- max(yy[tr_idx]) - ymin; if (yr == 0) yr <- 1
      rec$y <- list(min = ymin, range = yr)
      Ytr <- matrix((yy[tr_idx] - ymin) / yr, ncol = 1, dimnames = list(NULL, "Y1")); yte <- yy[-tr_idx]
    } else {
      yy <- factor(trimws(as.character(dfa[[target]])), levels = base$lev)
      Ytr <- if (length(base$lev) == 2) matrix(as.numeric(yy[tr_idx] == base$lev[2]), ncol = 1, dimnames = list(NULL, "Y1")) else {
        M <- sapply(base$lev, function(l) as.numeric(yy[tr_idx] == l)); colnames(M) <- paste0("Y", seq_along(base$lev)); M }
      yte <- yy[-tr_idx]
    }
    p <- list(Xtr = Xtr, Ytr = Ytr, type = base$type)
    tr <- nl_train(p, hidden, act, algorithm, lr, threshold, stepmax, 1, "auto", seed, TRUE, 3)
    sc <- c(sc, score_fn(list(rec = rec, train = tr), Xte, yte)); st <- c(st, tr$steps); se <- c(se, tr$secs)
  }
  c(score = mean(sc), steps = mean(st), secs = sum(se))
}


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO DATASETS  (robust loader)
# ─────────────────────────────────────────────────────────────────────────────

make_agri_data <- function() {
  set.seed(42); n <- 400
  rainfall <- runif(n, 300, 1200); temperature <- runif(n, 15, 38)
  humidity <- runif(n, 40, 95);    soil_ph <- runif(n, 5.0, 8.5)
  nitrogen <- runif(n, 10, 120);   phosphorus <- runif(n, 5, 80)
  potassium <- runif(n, 10, 100);  solar_rad <- runif(n, 150, 350)
  irrigation <- rbinom(n, 1, 0.6)
  soil <- sample(c("Loam", "Clay", "Sandy"), n, TRUE, prob = c(.5, .3, .2))
  yield <- 2.5 + 0.003 * rainfall + 0.08 * temperature - 0.001 * temperature^2 +
    0.02 * humidity - 0.3 * abs(soil_ph - 6.5) + 0.04 * nitrogen + 0.02 * phosphorus +
    0.015 * potassium + 0.005 * solar_rad + 1.2 * irrigation +
    ifelse(soil == "Loam", 0.6, ifelse(soil == "Clay", 0.1, -0.4)) + rnorm(n, 0, 0.5)
  data.frame(Rainfall_mm = round(rainfall, 1), Temperature_C = round(temperature, 1),
             Humidity_pct = round(humidity, 1), Soil_pH = round(soil_ph, 2),
             Nitrogen_kg = round(nitrogen, 1), Phosphorus_kg = round(phosphorus, 1),
             Potassium_kg = round(potassium, 1), Solar_Rad = round(solar_rad, 1),
             Irrigation = irrigation, Soil_Type = soil, Yield_t_ha = round(pmax(.5, yield), 3))
}

DEMOS <- list(
  iris    = list(label = "Iris flowers", type = "classification", target = "Species",
                 desc = "150 obs · 4 features · 3 species"),
  cancer  = list(label = "Breast cancer (Wisconsin)", type = "classification", target = "Class",
                 desc = "683 obs · 9 features · benign / malignant"),
  pima    = list(label = "Pima Indians diabetes", type = "classification", target = "diabetes",
                 desc = "Diabetes test result (pos / neg) from clinical measurements"),
  glass   = list(label = "Glass identification", type = "classification", target = "Type",
                 desc = "214 obs · 9 features · 6 glass types"),
  agri    = list(label = "Crop yield (simulated)", type = "regression", target = "Yield_t_ha",
                 desc = "400 obs · 10 features incl. categorical soil"),
  boston  = list(label = "Boston housing", type = "regression", target = "medv",
                 desc = "506 obs · 13 features · median house value"),
  mtcars  = list(label = "Motor Trend cars", type = "regression", target = "mpg",
                 desc = "32 obs · 10 features · fuel economy (tiny — overfits easily)")
)

# Fetch a packaged dataset in a way that works across package versions
# (data() into an env, then lazy-data export). Returns NULL unless it is a
# real data.frame — the v7 Pima bug came from silently getting NULL here.
pkg_data <- function(name, pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) return(NULL)
  e <- new.env()
  d <- tryCatch(suppressWarnings({
    utils::data(list = name, package = pkg, envir = e)
    get0(name, envir = e, inherits = FALSE)
  }), error = function(err) NULL)
  if (!is.data.frame(d)) d <- tryCatch(getExportedValue(pkg, name), error = function(err) NULL)
  if (is.data.frame(d)) d else NULL
}

load_demo <- function(key) {
  need_pkg <- function(d, pkg) if (is.null(d)) stop("Demo needs package '", pkg, "': install.packages('", pkg, "')") else d
  d <- switch(key,
              iris   = datasets::iris,
              cancer = { b <- need_pkg(pkg_data("BreastCancer", "mlbench"), "mlbench")[, -1]
              for (j in 1:9) b[[j]] <- to_num(b[[j]])
              b$Class <- as.character(b$Class); b[stats::complete.cases(b), ] },
              pima   = {
                p <- pkg_data("PimaIndiansDiabetes", "mlbench")
                if (is.null(p) || !"diabetes" %in% names(p)) {
                  # Fallback that ships with every R installation (package MASS)
                  a <- pkg_data("Pima.tr", "MASS"); b <- pkg_data("Pima.te", "MASS")
                  p <- need_pkg(if (!is.null(a) && !is.null(b)) rbind(a, b), "MASS")
                  names(p)[names(p) == "type"] <- "diabetes"
                  p$diabetes <- ifelse(p$diabetes == "Yes", "pos", "neg")
                }
                p$diabetes <- as.character(p$diabetes); p },
              glass  = { g <- need_pkg(pkg_data("Glass", "mlbench"), "mlbench")
              g$Type <- paste0("Type ", as.character(g$Type)); g },
              agri   = make_agri_data(),
              boston = need_pkg(pkg_data("Boston", "MASS"), "MASS"),
              mtcars = datasets::mtcars,
              stop("Unknown demo dataset"))
  d <- as.data.frame(d, stringsAsFactors = FALSE)
  for (j in names(d)) if (is.factor(d[[j]])) d[[j]] <- as.character(d[[j]])
  rownames(d) <- NULL
  if (!is.data.frame(d) || nrow(d) < 15) stop("Demo dataset '", key, "' could not be loaded.")
  d
}

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR SYSTEM
#  Plots: colour-blind-validated categorical palette in fixed order (never
#  cycled), single-hue sequential ramps, blue↔red diverging with grey midpoint.
#  Chrome (navbar, heroes, tiles): bold gradients from the chosen theme.
# ─────────────────────────────────────────────────────────────────────────────

SERIES  <- c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
OTHER   <- "#9CA3AF"
DIV_POS <- "#2a78d6"; DIV_NEG <- "#e34948"; DIV_MID <- "#e9e7e2"
STATUS  <- c(good = "#0ca30c", warn = "#fab219", bad = "#d03b3b", info = "#2a78d6")

THEMES <- list(
  aurora = list(name = "Aurora",  g = c("#4F46E5", "#7C3AED", "#0EA5E9", "#14B8A6"), seq = "#2a78d6"),
  sunset = list(name = "Sunset",  g = c("#F97316", "#EF4444", "#EC4899", "#8B5CF6"), seq = "#eb6834"),
  ocean  = list(name = "Ocean",   g = c("#0B3B66", "#1D4ED8", "#0891B2", "#22D3EE"), seq = "#2a78d6"),
  forest = list(name = "Forest",  g = c("#14532D", "#15803D", "#0D9488", "#84CC16"), seq = "#1baf7a"),
  candy  = list(name = "Candy",   g = c("#DB2777", "#C026D3", "#7C3AED", "#6366F1"), seq = "#6250d6")
)

mix_col <- function(a, b, t) {
  x <- grDevices::col2rgb(a) * (1 - t) + grDevices::col2rgb(b) * t
  grDevices::rgb(x[1], x[2], x[3], maxColorValue = 255)
}
seq_ramp <- function(hue, n = 9) grDevices::colorRampPalette(c(mix_col(hue, "white", .88), hue, mix_col(hue, "black", .5)))(n)
seq_scale <- function(hue) { r <- seq_ramp(hue, 5); lapply(seq_along(r), function(i) list((i - 1) / 4, r[i])) }
div_scale <- list(list(0, DIV_NEG), list(.5, DIV_MID), list(1, DIV_POS))
level_cols <- function(lev) {
  k <- length(lev); cols <- if (k <= 8) SERIES[seq_len(k)] else c(SERIES[1:7], rep(OTHER, k - 7))
  setNames(cols, lev)
}
alpha_col <- function(col, a) { x <- grDevices::col2rgb(col); sprintf("rgba(%d,%d,%d,%.2f)", x[1], x[2], x[3], a) }

# ─────────────────────────────────────────────────────────────────────────────
#  PARTIAL DEPENDENCE (PDP / ICE)
# ─────────────────────────────────────────────────────────────────────────────

pdp_data <- function(fit, var, grid_n = 25, base_n = 150, ice_n = 40, seed = 1) {
  rec <- fit$rec; d0 <- fit$raw_train
  set.seed(seed)
  base <- d0[sample.int(nrow(d0), min(base_n, nrow(d0))), , drop = FALSE]
  is_num <- var %in% names(rec$num)
  grid <- if (is_num) {
    v <- to_num(d0[[var]])
    unique(signif(stats::quantile(v, seq(.02, .98, length.out = grid_n), na.rm = TRUE, names = FALSE), 6))
  } else rec$cat[[var]]$levels
  nb <- nrow(base)
  big <- base[rep(seq_len(nb), times = length(grid)), , drop = FALSE]
  big[[var]] <- rep(grid, each = nb)
  out <- nl_predict_df(fit, big)
  gi <- rep(seq_along(grid), each = nb); id <- rep(seq_len(nb), times = length(grid))
  if (rec$type == "regression") { V <- matrix(out$pred, ncol = 1); series <- rec$target
  } else if (length(rec$lev) == 2) { V <- out$prob[, 2, drop = FALSE]; series <- paste0("P(", rec$lev[2], ")")
  } else { V <- out$prob; series <- paste0("P(", rec$lev, ")") }
  mean_df <- do.call(rbind, lapply(seq_len(ncol(V)), function(j) {
    agg <- tapply(V[, j], gi, function(x) c(mean(x), stats::quantile(x, c(.1, .9))))
    data.frame(grid = grid, series = series[j],
               mean = sapply(agg, `[`, 1), lo = sapply(agg, `[`, 2), hi = sapply(agg, `[`, 3),
               stringsAsFactors = FALSE)
  }))
  ice <- if (ncol(V) == 1) {
    k <- id <= min(ice_n, nb)
    data.frame(grid = grid[gi[k]], id = id[k], value = V[k, 1], stringsAsFactors = FALSE)
  } else NULL
  list(var = var, is_num = is_num, grid = grid, mean = mean_df, ice = ice,
       ylab = if (rec$type == "regression") paste("Predicted", rec$target) else "Predicted probability")
}

# ─────────────────────────────────────────────────────────────────────────────
#  AI INSIGHT ENGINE  (rule-based statistical interpretation)
# ─────────────────────────────────────────────────────────────────────────────

f3 <- function(x, d = 3) ifelse(is.na(x), "—", formatC(x, digits = d, format = "f"))
pc <- function(x, d = 1) ifelse(is.na(x), "—", paste0(formatC(100 * x, digits = d, format = "f"), "%"))
grade_r2 <- function(x) if (is.na(x)) "undefined" else if (x >= .9) "excellent" else if (x >= .75) "good" else if (x >= .5) "moderate" else if (x >= .25) "weak" else "poor"
grade_kappa <- function(k) if (is.na(k)) "undefined" else if (k > .8) "almost perfect" else if (k > .6) "substantial" else if (k > .4) "moderate" else if (k > .2) "fair" else if (k > 0) "slight" else "no better than chance"
grade_auc <- function(a) if (is.na(a)) "undefined" else if (a >= .9) "excellent" else if (a >= .8) "good" else if (a >= .7) "fair" else "poor"
p_fmt <- function(p) if (is.na(p)) "—" else if (p < .001) "p < 0.001" else paste0("p = ", formatC(p, digits = 3, format = "f"))
listify <- function(x) if (length(x) <= 1) x else paste0(paste(x[-length(x)], collapse = ", "), " and ", x[length(x)])

driver_summary <- function(fit, vars) {
  do.call(rbind, lapply(vars, function(v) {
    pd <- tryCatch(pdp_data(fit, v, grid_n = 12, base_n = 100, ice_n = 0), error = function(e) NULL)
    if (is.null(pd)) return(NULL)
    m <- pd$mean; ser <- unique(m$series)
    if (length(ser) == 1) {
      eff <- diff(range(m$mean))
      dir <- if (pd$is_num) {
        r <- suppressWarnings(stats::cor(as.numeric(m$grid), m$mean, method = "spearman"))
        if (is.na(r) || abs(r) < .3) "non-monotonic" else if (r > 0) "increases" else "decreases"
      } else paste0("highest for '", m$grid[which.max(m$mean)], "', lowest for '", m$grid[which.min(m$mean)], "'")
    } else {
      rng <- tapply(m$mean, m$series, function(x) diff(range(x)))
      top <- names(which.max(rng)); eff <- max(rng)
      mm <- m[m$series == top, ]
      dir <- if (pd$is_num) {
        r <- suppressWarnings(stats::cor(as.numeric(mm$grid), mm$mean, method = "spearman"))
        paste0(if (is.na(r) || abs(r) < .3) "changes non-monotonically" else if (r > 0) "raises" else "lowers", " ", top)
      } else paste0("'", mm$grid[which.max(mm$mean)], "' most favours ", top)
    }
    data.frame(Variable = v, Direction = dir, Effect = eff, stringsAsFactors = FALSE)
  }))
}

nl_insights <- function(fit) {
  rec <- fit$rec; tr <- fit$train; type <- rec$type
  S <- list(); flags <- list(); recs <- character()
  add <- function(id, title, icon, status, paras = character(), bullets = character())
    S[[id]] <<- list(title = title, icon = icon, status = status, paras = paras, bullets = bullets)
  n_tr <- fit$n[["train"]]; n_te <- fit$n[["test"]]
  n_par <- length(unlist(tr$model$weights[[tr$best]]))
  perm <- fit$perm; if (!is.null(perm)) perm <- perm[order(-perm$Importance), ]
  
  # ── Data ──
  raw <- fit$raw_train; miss <- sum(is.na(rbind(fit$raw_train, fit$raw_test)[rec$vars]))
  dpar <- sprintf("The model used %d rows (%d for training, %d held out for testing) and %d input variable(s), expanded to %d network inputs after encoding.",
                  n_tr + n_te, n_tr, n_te, length(rec$vars), length(rec$features))
  db <- character()
  if (length(rec$cat)) db <- c(db, paste0("Categorical predictors one-hot encoded: ", listify(names(rec$cat)), "."))
  if (miss > 0) db <- c(db, sprintf("%d missing predictor value(s) were imputed with training medians / modes.", miss))
  if (length(rec$dropped)) db <- c(db, paste0("Excluded automatically: ", paste(rec$dropped, collapse = "; "), "."))
  db <- c(db, paste0("Numeric predictors were scaled (", c(minmax = "min–max to [0,1]", zscore = "z-score", none = "no scaling")[[rec$scaling]],
                     ") using training-set statistics only, so no information leaked from the test set."))
  dstat <- "info"
  if (type == "classification") {
    tab <- table(fit$trainset$actual); ratio <- max(tab) / min(tab)
    db <- c(db, paste0("Class distribution (training): ", paste(sprintf("%s %d (%s)", names(tab), as.integer(tab), pc(tab / sum(tab))), collapse = ", "), "."))
    if (ratio > 3) {
      dstat <- "warn"; flags$imbalance <- TRUE
      db <- c(db, sprintf("Classes are imbalanced (largest/smallest = %.1f). Accuracy can look good simply by favouring the majority class — judge the model by balanced accuracy, recall per class and Kappa.", ratio))
      recs <- c(recs, "Because the classes are imbalanced, report balanced accuracy and per-class recall, and consider collecting more minority-class cases.")
    }
  }
  if (n_te < 30) {
    dstat <- "warn"; flags$small_test <- TRUE
    db <- c(db, sprintf("Only %d test observations — test metrics carry wide uncertainty.", n_te))
    recs <- c(recs, "With a small test set, confirm performance using 5-fold cross-validation in the Tuning tab.")
  }
  add("data", "Data & preprocessing", "🗂️", dstat, dpar, db)
  
  # ── Performance ──
  if (type == "regression") {
    m <- fit$test$metrics; mt <- fit$trainset$metrics
    a <- fit$test$actual; p <- fit$test$pred
    rmse0 <- sqrt(mean((a - mean(fit$trainset$actual))^2)); gain <- 1 - m[["RMSE"]] / rmse0
    g <- grade_r2(m[["R2"]])
    st <- if (m[["R2"]] >= .75) "good" else if (m[["R2"]] >= .5) "warn" else "bad"
    perf_p <- sprintf("On unseen test data the network explains %s of the variance in %s (R² = %s, %s fit). Typical prediction error is %s (RMSE) or %s on average (MAE)%s.",
                      pc(max(m[["R2"]], 0)), rec$target, f3(m[["R2"]]), g, signif(m[["RMSE"]], 4), signif(m[["MAE"]], 4),
                      if (!is.na(m[["MAPE"]])) paste0(", i.e. about ", formatC(m[["MAPE"]], digits = 1, format = "f"), "% of the observed value") else "")
    pb <- c(sprintf("Compared with a naive model that always predicts the training mean (RMSE %s), the network reduces error by %s.", signif(rmse0, 4), pc(gain)),
            sprintf("Correlation between predicted and observed values: r = %s.", f3(m[["r"]])))
    headline <- sprintf("an R² of %s (%s) and RMSE of %s", f3(m[["R2"]]), g, signif(m[["RMSE"]], 4))
    gap <- mt[["R2"]] - m[["R2"]]; train_score <- mt[["R2"]]; test_score <- m[["R2"]]; sname <- "R²"
    over <- !is.na(gap) && gap > .15; under <- !is.na(mt[["R2"]]) && mt[["R2"]] < .5
  } else {
    o <- fit$test$metrics$overall; ot <- fit$trainset$metrics$overall
    a <- fit$test$actual; n <- length(a); k <- sum(fit$test$pred == a)
    nir <- max(table(a)) / n
    pval <- stats::pbinom(k - 1, n, nir, lower.tail = FALSE)
    gk <- grade_kappa(o[["Kappa"]])
    st <- if (o[["Kappa"]] > .6 && pval < .05) "good" else if (o[["Kappa"]] > .4) "warn" else "bad"
    perf_p <- sprintf("The network classifies %s of unseen test cases correctly (%d of %d). Cohen's κ = %s indicates %s agreement beyond chance (Landis & Koch), and balanced accuracy is %s.",
                      pc(o[["Accuracy"]]), k, n, f3(o[["Kappa"]]), gk, pc(o[["BalAcc"]]))
    pb <- c(sprintf("No-information rate (always predicting the most common class): %s. One-sided binomial test that accuracy exceeds it: %s%s.",
                    pc(nir), p_fmt(pval), if (pval < .05) " — the model is significantly better than guessing" else " — NOT significantly better than guessing"),
            sprintf("Macro-averaged F1 = %s.", f3(o[["MacroF1"]])))
    if (!is.na(o[["AUC"]])) pb <- c(pb, sprintf("ROC AUC = %s (%s discrimination): a randomly chosen '%s' case receives a higher score than a randomly chosen '%s' case %s of the time.",
                                                f3(o[["AUC"]]), grade_auc(o[["AUC"]]), rec$lev[2], rec$lev[1], pc(o[["AUC"]], 0)))
    headline <- sprintf("%s test accuracy (κ = %s, %s agreement)", pc(o[["Accuracy"]]), f3(o[["Kappa"]]), gk)
    gap <- ot[["Accuracy"]] - o[["Accuracy"]]; train_score <- ot[["Accuracy"]]; test_score <- o[["Accuracy"]]; sname <- "accuracy"
    se_te <- sqrt(max(o[["Accuracy"]] * (1 - o[["Accuracy"]]), .01) / n)   # sampling noise of test accuracy
    over <- gap > max(.08, 2 * se_te); under <- ot[["BalAcc"]] < .7
    if (pval >= .05) flags$no_better <- TRUE
  }
  add("perf", "Predictive performance", "🎯", st, perf_p, pb)
  
  # ── Generalisation ──
  gp <- sprintf("Training %s is %s versus %s on the test set (gap %s).", sname,
                if (type == "regression") f3(train_score) else pc(train_score),
                if (type == "regression") f3(test_score) else pc(test_score),
                if (type == "regression") f3(gap) else pc(gap))
  ratio_p <- n_tr / n_par
  gb <- sprintf("The network has %d trainable weights for %d training rows (%.1f rows per weight).", n_par, n_tr, ratio_p)
  gstat <- "good"
  if (over) {
    gstat <- "warn"; flags$overfit <- TRUE
    gp <- paste(gp, "This sizeable gap suggests over-fitting: the network has partly memorised the training data.")
    recs <- c(recs, "Reduce over-fitting: use fewer neurons or a single hidden layer, raise the stop threshold (earlier stopping), or add more data.")
  } else if (under) {
    gstat <- "warn"; flags$underfit <- TRUE
    gp <- paste(gp, "Performance is limited even on the training data, which points to under-fitting (too simple a model or training stopped too early).")
    recs <- c(recs, "Address under-fitting: add neurons or a second hidden layer, lower the stop threshold, raise Max steps, or try the Tanh activation.")
  } else gp <- paste(gp, "Training and test performance are close, so the model generalises well to new data.")
  if (ratio_p < 5) {
    gb <- c(gb, if (gstat == "good") "Fewer than 5 rows per weight — the network is large relative to the data. It generalises acceptably here, but a smaller network would be a safer choice."
            else "Fewer than 5 rows per weight — the architecture is large relative to the data; a smaller network is likely to generalise better.")
  }
  add("gen", "Generalisation (over- / under-fitting)", "🧪", gstat, gp, gb)
  
  # ── Drivers ──
  drv <- NULL
  if (!is.null(perm) && nrow(perm)) {
    top <- head(perm$Feature[perm$Importance > 0], 3)
    drv <- if (length(top)) driver_summary(fit, top) else NULL
    imp_word <- if (type == "regression") "increase in test RMSE" else "drop in test accuracy"
    dp <- if (length(top)) sprintf("Permutation importance (the %s when a variable is randomly shuffled) identifies %s as the most influential predictor%s.",
                                   imp_word, listify(paste0("'", top, "'")), if (length(top) > 1) "s" else "")
    else "No single variable produced a measurable loss when shuffled — the model may be relying on combinations of weak signals."
    dbul <- character()
    if (!is.null(drv)) for (i in seq_len(nrow(drv))) {
      what <- paste("predicted", if (type == "regression") rec$target else "probability")
      txt <- switch(drv$Direction[i],
                    increases = paste(what, "increases as it rises"),
                    decreases = paste(what, "decreases as it rises"),
                    `non-monotonic` = paste("non-monotonic (rise-and-fall) effect on the", what),
                    drv$Direction[i])
      dbul <- c(dbul, sprintf("%s: %s (partial-dependence range %s).", drv$Variable[i], txt, signif(drv$Effect[i], 3)))
    }
    weak <- perm$Feature[perm$Importance <= 0]
    if (length(weak)) {
      dbul <- c(dbul, paste0("No measurable contribution on the test set: ", listify(weak), "."))
      recs <- c(recs, paste0("Consider re-training without low-value predictors (", listify(head(weak, 4)), ") — simpler models often generalise better."))
    }
    add("drivers", "What drives the predictions", "🔑", "info", dp, gsub("  +", " ", dbul))
  }
  
  # ── Error analysis ──
  if (type == "regression") {
    r <- fit$test$actual - fit$test$pred; p <- fit$test$pred
    bias <- mean(r); rb <- bias / stats::sd(fit$test$actual)
    het <- suppressWarnings(stats::cor(abs(r), p, method = "spearman"))
    outl <- sum(abs(r) > 3 * stats::sd(r))
    sw <- if (length(r) >= 3 && length(r) <= 5000) tryCatch(stats::shapiro.test(r)$p.value, error = function(e) NA) else NA
    eb <- c(sprintf("Mean residual (bias) = %s; %s.", signif(bias, 3),
                    if (abs(rb) > .1) paste("the model systematically", if (bias > 0) "under-predicts" else "over-predicts") else "no meaningful systematic bias"),
            sprintf("Residual spread vs prediction (Spearman ρ between |residual| and prediction) = %s%s.", f3(het),
                    if (!is.na(het) && het > .3) " — errors grow with the predicted value (heteroscedasticity)" else ""),
            sprintf("Shapiro–Wilk test of residual normality: %s.", p_fmt(sw)),
            sprintf("%d residual(s) exceed 3 standard deviations.", outl))
    est <- if (abs(rb) > .1 || (!is.na(het) && het > .3) || outl > 0) "warn" else "good"
    if (!is.na(het) && het > .3) recs <- c(recs, "Errors grow with the predicted value: try a log or square-root transform of the target before training.")
    add("errors", "Error analysis", "🔍", est, "Residuals (observed − predicted) on the test set were examined for bias, non-constant spread and outliers.", eb)
  } else {
    cm <- fit$test$metrics$cm; pcm <- fit$test$metrics$per_class
    off <- cm; diag(off) <- 0
    eb <- character()
    if (sum(off) > 0) {
      ix <- which(off == max(off), arr.ind = TRUE)[1, ]
      eb <- c(eb, sprintf("Most frequent confusion: actual '%s' predicted as '%s' (%d case%s).", rownames(cm)[ix[1]], colnames(cm)[ix[2]], max(off), if (max(off) > 1) "s" else ""))
    } else eb <- c(eb, "No misclassifications on the test set.")
    w <- pcm[which.min(pcm$Recall), ]
    eb <- c(eb, sprintf("Weakest class: '%s' (recall %s, precision %s, n = %d).", w$Class, pc(w$Recall), pc(w$Precision), w$Support))
    s <- pcm[which.max(pcm$F1), ]
    eb <- c(eb, sprintf("Best recognised class: '%s' (F1 = %s).", s$Class, f3(s$F1)))
    est <- if (min(pcm$Recall) < .6) "warn" else "good"
    if (min(pcm$Recall) < .6) recs <- c(recs, sprintf("Class '%s' is often missed — inspect the features that separate it and consider more examples of it.", w$Class))
    add("errors", "Error analysis", "🔍", est, "Test-set confusion matrix and per-class statistics were examined.", eb)
  }
  
  # ── Training ──
  tb <- c(sprintf("Architecture %s with %s activation, trained by %s on the %s error function.",
                  paste(c(length(rec$features), tr$hidden, ncol(tail(tr$model$weights[[tr$best]], 1)[[1]])), collapse = " → "),
                  ACT_LABELS[[tr$act]], tr$algorithm, toupper(tr$err)),
          sprintf("Converged after %s steps (stop threshold %s) in %.1f s; %d of %d repetition(s) converged.",
                  format(tr$steps, big.mark = ","), tr$threshold, tr$secs, tr$reps_ok, tr$reps_req))
  tstat <- "good"
  if (tr$threshold > tr$threshold_req) {
    tstat <- "warn"
    tb <- c(tb, sprintf("The requested threshold (%s) was not reached within Max steps, so it was relaxed automatically to %s.", tr$threshold_req, tr$threshold))
    recs <- c(recs, "Training needed a relaxed threshold — increase Max steps if you want a tighter fit.")
  }
  if (tr$reps_req == 1) recs <- c(recs, "Run 3–5 repetitions (different random starting weights) and keep the best, to reduce sensitivity to initialisation.")
  add("training", "Training & convergence", "⚙️", tstat, character(), tb)
  
  # ── Recommendations ──
  recs <- c(recs, "Use the Tuning tab (k-fold CV) to compare architectures objectively before finalising the model.",
            "Validate the final model on an independent data set before using it for decisions.")
  add("recs", "Recommendations", "💡", "info", character(), unique(recs))
  
  # ── Executive summary ──
  quality <- c(good = "performs well", warn = "performs moderately", bad = "performs poorly")[[st]]
  summ <- sprintf("A %s neural network (%s) was trained to predict '%s' from %d variable(s) in the %s data. It %s on held-out data, with %s.%s%s",
                  type, paste(tr$hidden, collapse = "-"), rec$target, length(rec$vars), fit$dataset %||% "supplied",
                  quality, headline,
                  if (!is.null(drv) && nrow(drv)) paste0(" The strongest driver is '", drv$Variable[1], "'.") else "",
                  if (isTRUE(flags$overfit)) " However, the gap between training and test performance indicates some over-fitting." else if (isTRUE(flags$underfit)) " Training performance is also limited, indicating under-fitting." else " Training and test performance agree, so the result should generalise.")
  S <- c(list(summary = list(title = "Executive summary", icon = "✨", status = st, paras = summ, bullets = character())), S)
  list(sections = S, drivers = drv, flags = flags, status = st)
}

# Optional: Claude (Anthropic API) narrative on top of the computed facts
claude_narrative <- function(fit, ins, api_key, model = "claude-sonnet-5-5") {
  if (!HAS_HTTR2) stop("Package 'httr2' is required: install.packages('httr2')")
  if (!nzchar(api_key)) stop("Enter an Anthropic API key (or set ANTHROPIC_API_KEY).")
  facts <- unlist(lapply(ins$sections, function(s) c(paste0("[", s$title, "]"), s$paras, paste("-", s$bullets))))
  prompt <- paste0(
    "You are an expert applied statistician reviewing a feed-forward neural network built in R (neuralnet). ",
    "Write an interpretation for a scientific report, for readers who are researchers but not ML specialists. ",
    "Use ONLY the facts below; do not invent numbers. Be specific, balanced and practical. ",
    "Use exactly these headings, each starting with '## ': Overview; How good is the model?; What drives the predictions; Caveats and limitations; Recommended next steps. ",
    "Use short paragraphs and '-' bullets. Keep it under 450 words.\n\nFACTS:\n", paste(facts, collapse = "\n"))
  req <- httr2::request("https://api.anthropic.com/v1/messages") |>
    httr2::req_headers(`x-api-key` = api_key, `anthropic-version` = "2023-06-01", `content-type` = "application/json") |>
    httr2::req_body_json(list(model = model, max_tokens = 1500,
                              messages = list(list(role = "user", content = prompt)))) |>
    httr2::req_timeout(120) |>
    httr2::req_error(is_error = function(resp) FALSE)
  resp <- httr2::req_perform(req)
  body <- tryCatch(httr2::resp_body_json(resp), error = function(e) list())
  if (httr2::resp_status(resp) >= 400) stop(body$error$message %||% paste("HTTP", httr2::resp_status(resp)))
  txt <- paste(vapply(body$content, function(b) if (identical(b$type, "text")) b$text else "", ""), collapse = "\n")
  if (!nzchar(txt)) stop("Empty response from the API.")
  txt
}

md_lite_html <- function(txt) {
  lines <- strsplit(txt, "\n", fixed = TRUE)[[1]]
  out <- character(); inlist <- FALSE
  for (ln in lines) {
    l <- htmltools::htmlEscape(trimws(ln))
    l <- gsub("\\*\\*(.+?)\\*\\*", "<b>\\1</b>", l, perl = TRUE)
    if (grepl("^#{1,3} ", l)) {
      if (inlist) { out <- c(out, "</ul>"); inlist <- FALSE }
      out <- c(out, paste0("<h6 class='ai-h'>", sub("^#{1,3} ", "", l), "</h6>"))
    } else if (grepl("^[-*•] ", l)) {
      if (!inlist) { out <- c(out, "<ul>"); inlist <- TRUE }
      out <- c(out, paste0("<li>", sub("^[-*•] ", "", l), "</li>"))
    } else if (nzchar(l)) {
      if (inlist) { out <- c(out, "</ul>"); inlist <- FALSE }
      out <- c(out, paste0("<p>", l, "</p>"))
    }
  }
  if (inlist) out <- c(out, "</ul>")
  HTML(paste(out, collapse = "\n"))
}

# ─────────────────────────────────────────────────────────────────────────────
#  STATIC FIGURES (ggplot) — used in the Word report
# ─────────────────────────────────────────────────────────────────────────────

gg_alpha <- function(col, a) grDevices::adjustcolor(col, alpha.f = a)

theme_nl <- function(base = 11) {
  theme_minimal(base_size = base) +
    theme(plot.title = element_text(face = "bold", size = base + 3, colour = "#111827"),
          plot.subtitle = element_text(colour = "#6B7280", margin = margin(b = 8)),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = "#ECEAE4", linewidth = .4),
          axis.title = element_text(colour = "#374151"), axis.text = element_text(colour = "#52514e"),
          legend.position = "bottom", legend.title = element_text(face = "bold"),
          strip.text = element_text(face = "bold", colour = "#111827"),
          plot.background = element_rect(fill = "white", colour = NA),
          plot.margin = margin(10, 14, 10, 10))
}

gg_avp <- function(fit, hue) {
  d <- data.frame(a = fit$test$actual, p = fit$test$pred); d$err <- abs(d$p - d$a)
  ggplot(d, aes(a, p)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "#9CA3AF", linewidth = .7) +
    geom_smooth(method = "lm", formula = y ~ x, se = TRUE, colour = SERIES[2], fill = gg_alpha(SERIES[2], .15), linewidth = .9) +
    geom_point(aes(fill = err), shape = 21, size = 3, colour = "white", stroke = .6) +
    scale_fill_gradientn(colours = seq_ramp(hue, 5)[2:5], name = "|error|") +
    labs(title = "Actual vs predicted (test set)", subtitle = "Dashed line = perfect prediction; orange = fitted trend",
         x = paste("Observed", fit$rec$target), y = paste("Predicted", fit$rec$target)) + theme_nl()
}
gg_resid <- function(fit, hue) {
  d <- data.frame(p = fit$test$pred, r = fit$test$actual - fit$test$pred); s <- stats::sd(d$r)
  ggplot(d, aes(p, r)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = -2 * s, ymax = 2 * s, fill = gg_alpha(hue, .08)) +
    geom_hline(yintercept = 0, colour = DIV_NEG, linetype = "dashed") +
    geom_point(shape = 21, fill = hue, colour = "white", size = 2.7, stroke = .5) +
    geom_smooth(method = "loess", formula = y ~ x, se = FALSE, colour = SERIES[2], linewidth = .8) +
    labs(title = "Residuals vs predicted", subtitle = "Shaded band = ±2 SD; orange = loess trend",
         x = "Predicted", y = "Residual (observed − predicted)") + theme_nl()
}
gg_resid_hist <- function(fit, hue) {
  r <- fit$test$actual - fit$test$pred
  ggplot(data.frame(r = r), aes(r)) +
    geom_histogram(aes(y = after_stat(density)), bins = 20, fill = hue, colour = "white", alpha = .85) +
    stat_function(fun = stats::dnorm, args = list(mean = mean(r), sd = stats::sd(r)), colour = SERIES[2], linewidth = 1) +
    labs(title = "Residual distribution", subtitle = "Orange = normal curve with the same mean and SD", x = "Residual", y = "Density") + theme_nl()
}
gg_cm <- function(fit, hue) {
  cm <- fit$test$metrics$cm; rs <- pmax(rowSums(cm), 1)
  d <- as.data.frame(cm); names(d) <- c("Actual", "Predicted", "n")
  d$pct <- d$n / rs[as.character(d$Actual)]
  ggplot(d, aes(Predicted, Actual, fill = pct)) +
    geom_tile(colour = "white", linewidth = 2) +
    geom_text(aes(label = sprintf("%d\n%.0f%%", n, 100 * pct), colour = pct > .55), fontface = "bold", size = 4.2, show.legend = FALSE) +
    scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "#111827")) +
    scale_fill_gradientn(colours = seq_ramp(hue, 5), limits = c(0, 1), name = "Row %", labels = function(x) paste0(100 * x, "%")) +
    scale_y_discrete(limits = rev) + coord_equal() +
    labs(title = "Confusion matrix (test set)", subtitle = "Rows = actual class; cells show count and row %") + theme_nl()
}
gg_roc <- function(fit, hue) {
  y <- as.numeric(fit$test$actual == fit$rec$lev[2]); s <- fit$test$prob[, 2]
  th <- sort(unique(c(Inf, s, -Inf)), decreasing = TRUE)
  d <- data.frame(fpr = sapply(th, function(t) mean(s[y == 0] >= t)), tpr = sapply(th, function(t) mean(s[y == 1] >= t)))
  ggplot(d, aes(fpr, tpr)) + geom_area(fill = gg_alpha(hue, .18)) +
    geom_abline(linetype = "dashed", colour = "#9CA3AF") + geom_line(colour = hue, linewidth = 1.3) +
    coord_equal() + labs(title = "ROC curve", subtitle = sprintf("AUC = %.3f", fit$test$metrics$overall["AUC"]),
                         x = "False positive rate", y = "True positive rate") + theme_nl()
}
gg_classbars <- function(fit) {
  pcm <- fit$test$metrics$per_class
  d <- data.frame(Class = rep(pcm$Class, 3), Metric = factor(rep(c("Precision", "Recall", "F1"), each = nrow(pcm)), levels = c("Precision", "Recall", "F1")),
                  v = c(pcm$Precision, pcm$Recall, pcm$F1))
  ggplot(d, aes(Class, v, fill = Metric)) + geom_col(position = position_dodge(.8), width = .72, colour = "white", linewidth = .6) +
    scale_fill_manual(values = SERIES[1:3]) + scale_y_continuous(limits = c(0, 1.05), labels = function(x) paste0(100 * x, "%")) +
    labs(title = "Per-class precision, recall and F1", x = NULL, y = NULL, fill = NULL) + theme_nl()
}
gg_importance <- function(df, title, xlab, hue) {
  if (is.null(df) || !nrow(df)) return(NULL)
  df <- df[order(df$Importance), ]; df$Feature <- factor(df$Feature, levels = df$Feature)
  df$pos <- df$Importance > 0
  ggplot(df, aes(Importance, Feature)) +
    geom_segment(aes(x = 0, xend = Importance, yend = Feature, colour = pos), linewidth = 1.6, lineend = "round") +
    geom_point(aes(colour = pos), size = 4) +
    geom_vline(xintercept = 0, colour = "#9CA3AF") +
    scale_colour_manual(values = c(`TRUE` = hue, `FALSE` = OTHER), guide = "none") +
    labs(title = title, x = xlab, y = NULL) + theme_nl()
}
gg_pdp <- function(fit, vars) {
  pds <- lapply(vars, function(v) tryCatch(pdp_data(fit, v, grid_n = 20, base_n = 120, ice_n = 0), error = function(e) NULL))
  pds <- Filter(Negate(is.null), pds); if (!length(pds)) return(NULL)
  d <- do.call(rbind, lapply(pds, function(p) {
    m <- p$mean; m$var <- p$var; m$x <- if (p$is_num) as.numeric(m$grid) else as.numeric(factor(m$grid, levels = p$grid)); m }))
  cols <- level_cols(unique(d$series))
  ggplot(d, aes(x, mean, colour = series, fill = series, group = series)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = .14, colour = NA) +
    geom_line(linewidth = 1.2) + geom_point(size = 1.6) +
    facet_wrap(~var, scales = "free_x") +
    scale_colour_manual(values = cols) + scale_fill_manual(values = cols) +
    labs(title = "Partial dependence of the top predictors",
         subtitle = "Average prediction as one variable changes; band = 10–90% of individual curves",
         x = NULL, y = pds[[1]]$ylab, colour = NULL, fill = NULL) +
    theme_nl() + theme(legend.position = if (length(cols) > 1) "bottom" else "none")
}
net_layout <- function(fit, show_bias = FALSE) {
  tr <- fit$train; W <- tr$model$weights[[tr$best]]; rec <- fit$rec
  sizes <- c(nrow(W[[1]]) - 1, sapply(W, ncol)); L <- length(sizes)
  out_lab <- if (rec$type == "regression") rec$target else if (length(rec$lev) == 2) paste0("P(", rec$lev[2], ")") else rec$lev
  labels <- c(list(rec$features), lapply(seq_len(L - 2), function(i) paste0("H", i, ".", seq_len(sizes[i + 1]))), list(out_lab))
  ypos <- function(n) if (n == 1) 0 else seq(1, -1, length.out = n) * min(1, n / 10 + .25)
  nodes <- do.call(rbind, lapply(seq_len(L), function(l) data.frame(
    layer = l, idx = seq_len(sizes[l]), x = l, y = ypos(sizes[l]), label = labels[[l]],
    kind = if (l == 1) "Input" else if (l == L) "Output" else "Hidden", stringsAsFactors = FALSE)))
  if (show_bias) nodes <- rbind(nodes, data.frame(layer = seq_len(L - 1), idx = 0, x = seq_len(L - 1) + .5,
                                                  y = max(nodes$y) + .28, label = paste0("Bias → L", 2:L), kind = "Bias"))
  ytop <- max(nodes$y)
  edges <- do.call(rbind, lapply(seq_along(W), function(i) {
    w <- W[[i]]; from_rows <- if (show_bias) 0:(nrow(w) - 1) else 1:(nrow(w) - 1)
    g <- expand.grid(fr = from_rows, to = seq_len(ncol(w)))
    g$w <- w[cbind(g$fr + 1, g$to)]
    src <- nodes[nodes$layer == i & nodes$kind != "Bias", ]; dst <- nodes[nodes$layer == i + 1 & nodes$kind != "Bias", ]
    g$x0 <- ifelse(g$fr == 0, i + .5, i); g$y0 <- ifelse(g$fr == 0, ytop, src$y[match(g$fr, src$idx)])
    g$x1 <- i + 1; g$y1 <- dst$y[g$to]
    g$from <- ifelse(g$fr == 0, "Bias", src$label[match(g$fr, src$idx)]); g$tolab <- dst$label[g$to]; g$layer <- i
    g
  }))
  list(nodes = nodes, edges = edges[is.finite(edges$y0), ], L = L, W = W)
}
gg_network <- function(fit) {
  nl <- net_layout(fit, FALSE); e <- nl$edges; n <- nl$nodes
  e <- e[order(abs(e$w)), ]; e$aw <- abs(e$w) / max(abs(e$w))
  e$sign <- ifelse(e$w >= 0, "positive", "negative")
  ggplot() +
    geom_segment(data = e, aes(x0, y0, xend = x1, yend = y1, colour = sign, alpha = aw, linewidth = aw)) +
    geom_point(data = n, aes(x, y, fill = kind), shape = 21, size = 6, colour = "white", stroke = 1.2) +
    geom_text(data = n[n$kind == "Input", ], aes(x - .06, y, label = label), hjust = 1, size = 3, colour = "#374151") +
    geom_text(data = n[n$kind == "Output", ], aes(x + .06, y, label = label), hjust = 0, size = 3.2, fontface = "bold", colour = "#111827") +
    scale_colour_manual(values = c(positive = DIV_POS, negative = DIV_NEG), name = "Weight") +
    scale_fill_manual(values = c(Input = SERIES[7], Hidden = SERIES[3], Output = SERIES[2]), name = "Layer") +
    scale_alpha(range = c(.15, .9), guide = "none") + scale_linewidth(range = c(.2, 2.2), guide = "none") +
    scale_x_continuous(expand = expansion(add = c(1.1, 1))) +
    labs(title = "Network architecture", subtitle = "Line width and opacity ∝ |weight|; blue = positive, red = negative") +
    theme_void(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 14), plot.subtitle = element_text(colour = "#6B7280"),
          legend.position = "bottom", plot.background = element_rect(fill = "white", colour = NA), plot.margin = margin(10, 10, 10, 10))
}
gg_tune <- function(tune, hue) {
  if (is.null(tune)) return(NULL)
  r <- tune[!is.na(tune$Score), ]; if (!nrow(r)) return(NULL)
  r$Config <- paste0(r$Layers, "L · ", r$Activation, ifelse(is.na(r$LR), "", paste0(" · lr ", r$LR)))
  ggplot(r, aes(Config, factor(Neurons), fill = Score)) + geom_tile(colour = "white", linewidth = 2) +
    geom_text(aes(label = formatC(Score, digits = 3, format = "g")), size = 3.6, fontface = "bold") +
    scale_fill_gradientn(colours = if (attr(tune, "higher")) seq_ramp(hue, 5)[1:4] else rev(seq_ramp(hue, 5)[1:4]), name = attr(tune, "metric")) +
    labs(title = "Hyper-parameter search", subtitle = paste(attr(tune, "metric"), if (attr(tune, "higher")) "(higher = better)" else "(lower = better)"),
         x = NULL, y = "Neurons per layer") + theme_nl() + theme(axis.text.x = element_text(angle = 25, hjust = 1))
}

# ─────────────────────────────────────────────────────────────────────────────
#  WORD REPORT (officer + flextable)
# ─────────────────────────────────────────────────────────────────────────────

save_png <- function(p, w = 7, h = 4.3) {
  f <- tempfile(fileext = ".png")
  if (HAS_RAGG) ragg::agg_png(f, width = w, height = h, units = "in", res = 170) else grDevices::png(f, width = w, height = h, units = "in", res = 170)
  print(p); grDevices::dev.off(); f
}

build_word_report <- function(file, fit, ins, ai_text = NULL, tune = NULL, theme = THEMES$aurora,
                              title = "Neural Network Analysis Report", author = "", sections = c("ai", "metrics", "figures", "interp", "tune", "appendix")) {
  if (!HAS_OFFICER) stop("Install the Word report packages: install.packages(c('officer','flextable'))")
  op <- officer::fp_text; ft <- officer::ftext
  c1 <- theme$g[1]; c2 <- theme$g[2]; hue <- theme$seq
  tint <- mix_col(c1, "white", .92)
  ftab <- function(df, digits = 4, full = TRUE) {
    num <- vapply(df, is.numeric, TRUE)
    df[num] <- lapply(df[num], function(x) signif(x, digits))
    x <- flextable::flextable(df)
    x <- flextable::theme_vanilla(x)
    x <- flextable::bg(x, part = "header", bg = c1)
    x <- flextable::color(x, part = "header", color = "white")
    x <- flextable::bold(x, part = "header")
    if (nrow(df) > 1) x <- flextable::bg(x, i = seq(2, nrow(df), 2), bg = tint, part = "body")
    x <- flextable::fontsize(x, size = 9.5, part = "all")
    x <- flextable::font(x, fontname = "Calibri", part = "all")
    x <- flextable::border_outer(x, border = officer::fp_border(color = c1, width = 1.2))
    x <- flextable::autofit(x)
    if (full) x <- flextable::set_table_properties(x, layout = "autofit", width = 1)
    x
  }
  callout <- function(doc, text, col) {
    x <- flextable::flextable(data.frame(t = text))
    x <- flextable::delete_part(x, "header")
    x <- flextable::bg(x, bg = mix_col(col, "white", .9))
    x <- flextable::border_remove(x)
    x <- flextable::border(x, border.left = officer::fp_border(color = col, width = 6))
    x <- flextable::font(x, fontname = "Calibri"); x <- flextable::fontsize(x, size = 10.5)
    x <- flextable::padding(x, padding = 8)
    x <- flextable::width(x, width = 6.5)
    flextable::body_add_flextable(doc, x)
  }
  h1 <- function(doc, t) officer::body_add_par(doc, t, style = "heading 1")
  h2 <- function(doc, t) officer::body_add_par(doc, t, style = "heading 2")
  par <- function(doc, t) officer::body_add_fpar(doc, officer::fpar(ft(t, op(font.family = "Calibri", font.size = 10.5, color = "#1F2937")),
                                                                    fp_p = officer::fp_par(padding.bottom = 4, line_spacing = 1.15)))
  bullets <- function(doc, b) { for (x in b) doc <- officer::body_add_fpar(doc, officer::fpar(ft("•  ", op(color = c2, bold = TRUE, font.size = 11)), ft(x, op(font.size = 10.5, font.family = "Calibri")),
                                                                                              fp_p = officer::fp_par(padding.left = 14, padding.bottom = 3))); doc }
  fig <- function(doc, p, w = 6.4, h = 3.9, caption = NULL) {
    if (is.null(p)) return(doc)
    f <- tryCatch(save_png(p, w, h), error = function(e) { warning("Figure skipped: ", conditionMessage(e)); NULL }); if (is.null(f)) return(doc)
    doc <- officer::body_add_img(doc, f, width = w, height = h, style = "centered")
    if (!is.null(caption)) doc <- officer::body_add_fpar(doc, officer::fpar(ft(caption, op(italic = TRUE, color = "#6B7280", font.size = 9)), fp_p = officer::fp_par(text.align = "center", padding.bottom = 10)))
    doc
  }
  stat_col <- c(good = "#0ca30c", warn = "#E8A200", bad = "#d03b3b", info = c2)
  stat_word <- c(good = "✔ Good", warn = "⚠ Attention", bad = "✖ Concern", info = "ℹ Note")
  rec <- fit$rec; tr <- fit$train
  
  doc <- officer::read_docx()
  # ── Cover banner ──
  sub <- sprintf("%s  ·  target: %s  ·  %s", fit$dataset %||% "Dataset", rec$target, tools::toTitleCase(rec$type))
  when <- paste0(if (nzchar(author)) paste0(author, "  ·  ") else "", format(Sys.time(), "%d %B %Y"))
  bn <- flextable::flextable(data.frame(x = c("NEUROLAB v8  ·  NEURAL NETWORK STUDIO", title, sub, when)))
  bn <- flextable::delete_part(bn, "header"); bn <- flextable::border_remove(bn)
  bn <- flextable::bg(bn, i = 1:2, bg = c1); bn <- flextable::bg(bn, i = 3:4, bg = c2)
  bn <- flextable::color(bn, color = "white")
  bn <- flextable::font(bn, fontname = "Calibri")
  bn <- flextable::fontsize(bn, i = 1, size = 9); bn <- flextable::bold(bn, i = 1:2)
  bn <- flextable::fontsize(bn, i = 2, size = 24); bn <- flextable::fontsize(bn, i = 3, size = 12); bn <- flextable::fontsize(bn, i = 4, size = 10)
  bn <- flextable::padding(bn, padding.left = 18, padding.right = 18, padding.top = 6, padding.bottom = 6)
  bn <- flextable::padding(bn, i = 1, padding.top = 16); bn <- flextable::padding(bn, i = 4, padding.bottom = 16)
  bn <- flextable::width(bn, width = 6.5)
  doc <- flextable::body_add_flextable(doc, bn)
  doc <- par(doc, "")
  # ── KPI strip ──
  kp <- if (rec$type == "regression") { m <- fit$test$metrics
  c(`R² (test)` = f3(m[["R2"]]), RMSE = format(signif(m[["RMSE"]], 4)), MAE = format(signif(m[["MAE"]], 4)), `Pearson r` = f3(m[["r"]]))
  } else { o <- fit$test$metrics$overall
  c(Accuracy = pc(o[["Accuracy"]]), `Cohen's κ` = f3(o[["Kappa"]]), `Balanced acc.` = pc(o[["BalAcc"]]),
    if (!is.na(o[["AUC"]])) c(`ROC AUC` = f3(o[["AUC"]])) else c(`Macro F1` = f3(o[["MacroF1"]]))) }
  kd <- as.data.frame(as.list(kp), check.names = FALSE)
  kcols <- SERIES[c(1, 3, 2, 7)][seq_along(kp)]
  kt <- flextable::flextable(kd); kt <- flextable::border_remove(kt)
  kt <- flextable::bg(kt, part = "header", bg = kcols); kt <- flextable::bg(kt, part = "body", bg = kcols)
  for (j in seq_along(kp)) { kt <- flextable::bg(kt, j = j, part = "all", bg = kcols[j]) }
  kt <- flextable::color(kt, part = "all", color = "white"); kt <- flextable::font(kt, part = "all", fontname = "Calibri")
  kt <- flextable::fontsize(kt, part = "header", size = 9); kt <- flextable::fontsize(kt, part = "body", size = 20)
  kt <- flextable::bold(kt, part = "all"); kt <- flextable::align(kt, part = "all", align = "center")
  kt <- flextable::border_inner_v(kt, border = officer::fp_border(color = "white", width = 4), part = "all")
  kt <- flextable::padding(kt, part = "all", padding = 8); kt <- flextable::width(kt, width = 6.5 / length(kp))
  doc <- flextable::body_add_flextable(doc, kt)
  doc <- par(doc, "")
  doc <- callout(doc, ins$sections$summary$paras, stat_col[[ins$status]])
  doc <- par(doc, "")
  
  # ── Key metrics ──
  if ("metrics" %in% sections) {
    doc <- h1(doc, "Model at a glance")
    cfg <- data.frame(Setting = c("Task", "Target", "Inputs (after encoding)", "Hidden layers", "Activation", "Algorithm", "Error function",
                                  "Stop threshold", "Steps to converge", "Train / test rows", "Trainable weights"),
                      Value = c(rec$type, rec$target, length(rec$features), paste(tr$hidden, collapse = " → "), ACT_LABELS[[tr$act]],
                                tr$algorithm, toupper(tr$err), as.character(tr$threshold), format(tr$steps, big.mark = ","),
                                paste(fit$n["train"], "/", fit$n["test"]), length(unlist(tr$model$weights[[tr$best]]))))
    doc <- h2(doc, "Configuration"); doc <- flextable::body_add_flextable(doc, ftab(cfg))
    doc <- h2(doc, "Performance (train vs test)")
    mt <- if (rec$type == "regression") {
      a <- fit$trainset$metrics; b <- fit$test$metrics
      data.frame(Metric = c("RMSE", "MAE", "R²", "Pearson r", "MAPE (%)"), Train = as.numeric(a), Test = as.numeric(b))
    } else {
      a <- fit$trainset$metrics$overall; b <- fit$test$metrics$overall
      d <- data.frame(Metric = c("Accuracy", "Cohen's κ", "Balanced accuracy", "Macro F1", "ROC AUC"), Train = as.numeric(a), Test = as.numeric(b))
      d[!is.na(d$Test), ]
    }
    doc <- flextable::body_add_flextable(doc, ftab(mt))
    if (rec$type == "classification") {
      doc <- h2(doc, "Per-class statistics (test)")
      doc <- flextable::body_add_flextable(doc, ftab(fit$test$metrics$per_class))
    }
  }
  
  # ── AI interpretation ──
  if ("ai" %in% sections) {
    doc <- officer::body_add_break(doc)
    doc <- h1(doc, "AI interpretation")
    doc <- officer::body_add_fpar(doc, officer::fpar(ft("Generated automatically by the NeuroLab insight engine from the fitted model's statistics.",
                                                        op(italic = TRUE, color = "#6B7280", font.size = 9.5))))
    for (s in ins$sections[names(ins$sections) != "summary"]) {
      doc <- officer::body_add_fpar(doc, officer::fpar(ft(paste0(s$title, "   "), op(bold = TRUE, font.size = 13, color = "#111827", font.family = "Calibri")),
                                                       ft(stat_word[[s$status]], op(bold = TRUE, font.size = 9.5, color = stat_col[[s$status]], font.family = "Calibri")),
                                                       fp_p = officer::fp_par(padding.top = 10, padding.bottom = 4)))
      for (p in s$paras) doc <- par(doc, p)
      if (length(s$bullets)) doc <- bullets(doc, s$bullets)
    }
    if (!is.null(ai_text) && nzchar(ai_text)) {
      doc <- h2(doc, "Narrative by Claude (Anthropic)")
      for (ln in strsplit(ai_text, "\n", fixed = TRUE)[[1]]) {
        l <- gsub("\\*\\*", "", trimws(ln)); if (!nzchar(l)) next
        if (grepl("^#{1,3} ", l)) doc <- officer::body_add_par(doc, sub("^#{1,3} ", "", l), style = "heading 3")
        else if (grepl("^[-*•] ", l)) doc <- bullets(doc, sub("^[-*•] ", "", l))
        else doc <- par(doc, l)
      }
    }
  }
  
  # ── Figures ──
  if ("figures" %in% sections) {
    doc <- officer::body_add_break(doc)
    doc <- h1(doc, "Diagnostic figures")
    if (rec$type == "regression") {
      doc <- fig(doc, gg_avp(fit, hue), caption = "Figure 1. Observed vs predicted values on the held-out test set.")
      doc <- fig(doc, gg_resid(fit, hue), caption = "Figure 2. Residuals against predictions.")
      doc <- fig(doc, gg_resid_hist(fit, hue), 6, 3.4, caption = "Figure 3. Distribution of test residuals.")
    } else {
      k <- length(rec$lev)
      doc <- fig(doc, gg_cm(fit, hue), min(6.4, 3 + .6 * k), min(5.5, 2.8 + .55 * k), caption = "Figure 1. Confusion matrix (test set).")
      if (k == 2) doc <- fig(doc, gg_roc(fit, hue), 5, 4.6, caption = "Figure 2. ROC curve (test set).")
      doc <- fig(doc, gg_classbars(fit), 6.4, 3.6, caption = paste0("Figure ", if (k == 2) 3 else 2, ". Per-class precision, recall and F1."))
    }
  }
  
  # ── Interpretation figures ──
  if ("interp" %in% sections) {
    doc <- officer::body_add_break(doc)
    doc <- h1(doc, "Model interpretation")
    doc <- fig(doc, gg_importance(fit$perm, "Permutation importance (test set)",
                                  if (rec$type == "regression") "Increase in RMSE when shuffled" else "Drop in accuracy when shuffled", hue),
               6.4, max(2.8, .32 * length(rec$vars) + 1.4), caption = "Variables whose shuffling hurts the model most are the most important.")
    top <- if (!is.null(fit$perm)) head(fit$perm$Feature[order(-fit$perm$Importance)], 4) else character()
    if (length(top)) doc <- fig(doc, gg_pdp(fit, top), 6.6, 4.6, caption = "Partial dependence: how the prediction responds to each top variable.")
    doc <- fig(doc, gg_importance(fit$olden, "Olden connection-weight importance", "Olden importance", hue),
               6.4, max(2.8, .3 * length(rec$features) + 1.4))
    doc <- fig(doc, gg_network(fit), 6.6, 4.6, caption = "Trained network. Inputs on the left, output(s) on the right.")
  }
  
  # ── Tuning ──
  if ("tune" %in% sections && !is.null(tune)) {
    doc <- officer::body_add_break(doc)
    doc <- h1(doc, "Hyper-parameter search")
    doc <- fig(doc, gg_tune(tune, hue), 6.6, 4)
    tt <- tune[order(tune$Score, decreasing = attr(tune, "higher"), na.last = TRUE), setdiff(names(tune), "Issue")]
    names(tt)[names(tt) == "Score"] <- attr(tune, "metric")
    doc <- flextable::body_add_flextable(doc, ftab(head(tt, 20)))
  }
  
  # ── Appendix ──
  if ("appendix" %in% sections) {
    doc <- officer::body_add_break(doc)
    doc <- h1(doc, "Appendix")
    doc <- h2(doc, "Network inputs")
    doc <- flextable::body_add_flextable(doc, ftab(data.frame(Input = rec$features, `Source variable` = rec$groups, check.names = FALSE)))
    if (!is.null(fit$perm)) {
      doc <- h2(doc, "Permutation importance values")
      pm <- fit$perm[order(-fit$perm$Importance), ]; doc <- flextable::body_add_flextable(doc, ftab(pm))
    }
    doc <- par(doc, "")
    doc <- officer::body_add_fpar(doc, officer::fpar(ft("Method notes: networks were fitted with the R package neuralnet. Predictors were scaled using training data only; regression targets were min–max scaled for training and back-transformed. Test metrics are computed on a stratified hold-out sample never seen during training. Permutation importance = mean change in the test metric over 5 random shuffles of each variable. Partial dependence curves average the model's prediction over observed training rows while varying one variable.",
                                                        op(font.size = 9, color = "#6B7280", font.family = "Calibri"))))
  }
  print(doc, target = file)
  invisible(file)
}

# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE PLOT HELPERS (plotly)
# ─────────────────────────────────────────────────────────────────────────────

pl <- function(p, xlab = NULL, ylab = NULL, legend = TRUE, ...) {
  ax <- list(gridcolor = "rgba(148,163,184,.16)", zerolinecolor = "rgba(148,163,184,.35)",
             linecolor = "rgba(148,163,184,.35)", tickfont = list(size = 11), automargin = TRUE)
  p |>
    layout(paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)",
           font = list(family = "Plus Jakarta Sans, system-ui, Segoe UI, Arial, sans-serif", color = "#8B95A7", size = 12),
           xaxis = c(ax, list(title = xlab)), yaxis = c(ax, list(title = ylab)),
           showlegend = legend, legend = list(orientation = "h", y = -0.2, font = list(size = 11)),
           margin = list(l = 10, r = 10, t = 36, b = 10), hovermode = "closest",
           hoverlabel = list(bgcolor = "#0F172A", bordercolor = "#0F172A", font = list(color = "white", size = 12)), ...) |>
    config(displaylogo = FALSE, toImageButtonOptions = list(format = "png", scale = 3),
           modeBarButtonsToRemove = c("lasso2d", "select2d", "autoScale2d"))
}

empty_plot <- function(msg) {
  plot_ly(type = "scatter", mode = "markers") |> layout(
    xaxis = list(visible = FALSE), yaxis = list(visible = FALSE),
    paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)",
    annotations = list(list(text = msg, showarrow = FALSE, font = list(size = 14, color = "#8B95A7")))) |>
    config(displayModeBar = FALSE)
}

network_plot <- function(fit, show_bias = TRUE, max_edges = 1500) {
  nl <- net_layout(fit, show_bias); edges <- nl$edges; nodes <- nl$nodes
  if (nrow(edges) > max_edges) edges <- edges[order(-abs(edges$w))[seq_len(max_edges)], ]
  q <- stats::quantile(abs(edges$w), c(.25, .5, .75))
  edges$bin <- findInterval(abs(edges$w), q) + 1
  widths <- c(.6, 1.4, 2.6, 4.4); ops <- c(.18, .35, .55, .85)
  p <- plot_ly(type = "scatter", mode = "lines")
  for (s in c("pos", "neg")) for (b in 1:4) {
    e <- edges[(edges$w >= 0) == (s == "pos") & edges$bin == b, ]; if (!nrow(e)) next
    p <- add_trace(p, x = as.vector(rbind(e$x0, e$x1, NA)), y = as.vector(rbind(e$y0, e$y1, NA)),
                   type = "scatter", mode = "lines", hoverinfo = "skip", showlegend = FALSE,
                   line = list(color = if (s == "pos") DIV_POS else DIV_NEG, width = widths[b], shape = "spline"), opacity = ops[b])
  }
  mid <- data.frame(x = (edges$x0 + edges$x1) / 2, y = (edges$y0 + edges$y1) / 2,
                    t = sprintf("<b>%s → %s</b><br>weight = %.4f", edges$from, edges$tolab, edges$w))
  p <- add_trace(p, data = mid, x = ~x, y = ~y, type = "scatter", mode = "markers", text = ~t,
                 hoverinfo = "text", marker = list(size = 7, opacity = 0), showlegend = FALSE)
  ncol <- c(Input = SERIES[7], Hidden = SERIES[3], Output = SERIES[2], Bias = OTHER)
  for (k in names(ncol)) {
    nd <- nodes[nodes$kind == k, ]; if (!nrow(nd)) next
    pos <- switch(k, Input = "middle left", Output = "middle right", Bias = "top center", "middle center")
    p <- add_trace(p, data = nd, x = ~x, y = ~y, type = "scatter", mode = "markers+text",
                   text = if (k %in% c("Input", "Output")) ~label else "", textposition = pos,
                   hovertext = ~paste0("<b>", label, "</b><br>", kind, " node"), hoverinfo = "text", name = k,
                   marker = list(size = if (k == "Bias") 12 else 24, color = ncol[[k]], line = list(color = "white", width = 3)),
                   textfont = list(size = 11))
  }
  pl(p, legend = TRUE) |>
    layout(xaxis = list(visible = FALSE, range = c(.15, nl$L + .85)),
           yaxis = list(visible = FALSE, range = c(-1.25, max(nodes$y) + .35)))
}

sankey_plot <- function(fit, max_links = 600) {
  nl <- net_layout(fit, FALSE); n <- nl$nodes; e <- nl$edges
  if (nrow(e) > max_links) e <- e[order(-abs(e$w))[seq_len(max_links)], ]
  key <- paste(n$layer, n$idx)
  e$s <- match(paste(e$layer, e$fr), key) - 1; e$t <- match(paste(e$layer + 1, e$to), key) - 1
  ncol <- c(Input = SERIES[7], Hidden = SERIES[3], Output = SERIES[2])[n$kind]
  plot_ly(type = "sankey", arrangement = "snap", valueformat = ".3f",
          node = list(label = n$label, color = unname(ncol), pad = 14, thickness = 16, line = list(color = "white", width = 1)),
          link = list(source = e$s, target = e$t, value = abs(e$w),
                      color = ifelse(e$w >= 0, alpha_col(DIV_POS, .38), alpha_col(DIV_NEG, .38)),
                      customdata = sprintf("%.4f", e$w),
                      hovertemplate = "%{source.label} → %{target.label}<br>weight = %{customdata}<extra></extra>")) |>
    pl(legend = FALSE)
}

importance_plot <- function(df, xlab, hue) {
  if (is.null(df) || !nrow(df)) return(empty_plot("Not available"))
  df <- df[order(df$Importance), ]; df$Feature <- factor(df$Feature, levels = df$Feature)
  col <- ifelse(df$Importance > 0, hue, OTHER)
  plot_ly(df) |>
    add_segments(x = 0, xend = ~Importance, y = ~Feature, yend = ~Feature, line = list(color = col, width = 4), hoverinfo = "skip", showlegend = FALSE) |>
    add_markers(x = ~Importance, y = ~Feature, marker = list(color = col, size = 13, line = list(color = "white", width = 2)),
                text = ~sprintf("<b>%s</b><br>%.4g", Feature, Importance), hoverinfo = "text", showlegend = FALSE) |>
    pl(xlab, NULL, legend = FALSE)
}

pdp_plot <- function(pd, hue, show_ice = TRUE) {
  m <- pd$mean; ser <- unique(m$series); p <- plot_ly(type = "scatter", mode = "lines")
  if (pd$is_num) {
    x <- as.numeric(m$grid)
    if (length(ser) == 1) {
      if (show_ice && !is.null(pd$ice)) {
        ic <- pd$ice[order(pd$ice$id, as.numeric(pd$ice$grid)), ]
        xs <- unlist(lapply(split(ic, ic$id), function(d) c(as.numeric(d$grid), NA)))
        ys <- unlist(lapply(split(ic, ic$id), function(d) c(d$value, NA)))
        p <- add_trace(p, x = xs, y = ys, type = "scatter", mode = "lines", name = "Individual (ICE)",
                       line = list(color = alpha_col(hue, .18), width = 1), hoverinfo = "skip")
      }
      p <- add_trace(p, x = c(x, rev(x)), y = c(m$hi, rev(m$lo)), type = "scatter", mode = "lines", fill = "toself",
                     fillcolor = alpha_col(hue, .14), line = list(width = 0), name = "10–90% band", hoverinfo = "skip")
      p <- add_trace(p, x = x, y = m$mean, type = "scatter", mode = "lines+markers", name = "Average (PDP)",
                     line = list(color = hue, width = 4, shape = "spline"), marker = list(size = 7, color = hue, line = list(color = "white", width = 2)),
                     hovertemplate = paste0(pd$var, " = %{x:.4g}<br>prediction = %{y:.4g}<extra></extra>"))
    } else {
      cols <- level_cols(ser)
      for (s in ser) { d <- m[m$series == s, ]
      p <- add_trace(p, x = as.numeric(d$grid), y = d$mean, type = "scatter", mode = "lines+markers", name = s,
                     line = list(color = cols[[s]], width = 3.5, shape = "spline"), marker = list(size = 6, color = cols[[s]]),
                     hovertemplate = paste0(s, "<br>", pd$var, " = %{x:.4g}<br>%{y:.3f}<extra></extra>")) }
    }
  } else {
    cols <- level_cols(ser)
    for (s in ser) { d <- m[m$series == s, ]
    p <- add_trace(p, x = d$grid, y = d$mean, type = "bar", name = s, marker = list(color = cols[[s]], line = list(color = "white", width = 2)),
                   error_y = list(type = "data", symmetric = FALSE, array = d$hi - d$mean, arrayminus = d$mean - d$lo, color = "#64748B"),
                   hovertemplate = paste0(s, "<br>%{x}: %{y:.4g}<extra></extra>")) }
    p <- layout(p, barmode = "group", bargap = .35)
  }
  pl(p, pd$var, pd$ylab)
}

# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────

hero_svg <- '<svg class="hero-net" viewBox="0 0 220 120" aria-hidden="true">
  <g class="edges"><line x1="20" y1="20" x2="110" y2="30"/><line x1="20" y1="20" x2="110" y2="90"/>
  <line x1="20" y1="60" x2="110" y2="30"/><line x1="20" y1="60" x2="110" y2="60"/><line x1="20" y1="100" x2="110" y2="60"/>
  <line x1="20" y1="100" x2="110" y2="90"/><line x1="110" y1="30" x2="200" y2="45"/><line x1="110" y1="60" x2="200" y2="45"/>
  <line x1="110" y1="60" x2="200" y2="80"/><line x1="110" y1="90" x2="200" y2="80"/></g>
  <g class="nodes"><circle cx="20" cy="20" r="7"/><circle cx="20" cy="60" r="7"/><circle cx="20" cy="100" r="7"/>
  <circle cx="110" cy="30" r="8"/><circle cx="110" cy="60" r="8"/><circle cx="110" cy="90" r="8"/>
  <circle cx="200" cy="45" r="9"/><circle cx="200" cy="80" r="9"/></g></svg>'

hero <- function(title, subtitle, cls = "hero-a", badge = NULL) {
  div(class = paste("hero", cls), div(class = "blob b1"), div(class = "blob b2"), HTML(hero_svg),
      div(class = "hero-text", if (!is.null(badge)) span(class = "hero-badge", badge), h2(title), p(subtitle)))
}

kpi <- function(label, value, icon, cls = "k1", sub = NULL, small = FALSE, num = NULL, dec = 0, suffix = "", prefix = "") {
  v <- if (!is.null(num) && is.finite(num))
    span(class = "kpi-value", `data-count` = num, `data-dec` = dec, `data-suffix` = suffix, `data-prefix` = prefix, value)
  else span(class = "kpi-value", value)
  div(class = paste("kpi", cls, if (small) "kpi-small"), title = paste(label, value),
      div(class = "kpi-icon", icon),
      div(class = "kpi-body", div(class = "kpi-label", label), v, if (!is.null(sub)) div(class = "kpi-sub", sub)))
}

ch <- function(icon, title, n = 1, ...) card_header(span(class = paste0("ch-ic ci", n), icon), span(class = "ch-t", title), ...)

theme_css <- paste(vapply(names(THEMES), function(k) {
  g <- THEMES[[k]]$g
  sprintf("%s{--g1:%s;--g2:%s;--g3:%s;--g4:%s;--grad:linear-gradient(115deg,%s 0%%,%s 35%%,%s 70%%,%s 100%%);--grad2:linear-gradient(120deg,%s,%s);--grad3:linear-gradient(120deg,%s,%s);--acc:%s;--acc-soft:%s;--acc-glow:%s}",
          if (k == "aurora") ":root,body.th-aurora" else paste0("body.th-", k),
          g[1], g[2], g[3], g[4], g[1], g[2], g[3], g[4], g[3], g[1], g[2], g[4], g[2], alpha_col(g[2], .12), alpha_col(g[2], .45))
}, ""), collapse = "\n")

APP_CSS <- paste0(theme_css, "
:root{--nl-radius:20px;--nl-shadow:0 1px 2px rgba(15,23,42,.04),0 10px 30px rgba(15,23,42,.07)}
body{background:radial-gradient(1100px 600px at 105% -10%,var(--acc-soft),transparent 60%),
      radial-gradient(900px 520px at -10% 110%,rgba(20,184,166,.10),transparent 60%),var(--bs-body-bg);background-attachment:fixed;transition:background .4s}
.navbar{background:var(--grad)!important;background-size:200% 200%!important;animation:gradShift 14s ease infinite;
  box-shadow:0 8px 30px -8px var(--acc-glow);border:0!important}
@keyframes gradShift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.navbar .navbar-brand,.navbar .nav-link{color:rgba(255,255,255,.88)!important;font-weight:650}
.navbar .navbar-nav .nav-link.active,.navbar .navbar-nav .nav-link:hover,.navbar .navbar-nav .show>.nav-link{color:#fff!important;background:rgba(255,255,255,.22)!important;border-radius:12px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.35)}
.navbar .nav-link{padding:.45rem .8rem!important;margin:0 2px;transition:all .2s}
.brand{display:flex;align-items:center;gap:.6rem;font-weight:800;letter-spacing:-.4px;color:#fff}
.brand-logo{width:36px;height:36px;border-radius:12px;background:rgba(255,255,255,.2);display:grid;place-items:center;
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.4);font-size:19px;animation:float 4s ease-in-out infinite}
@keyframes float{50%{transform:translateY(-3px) rotate(-4deg)}}
.brand small{font-weight:700;font-size:.7rem;margin-left:.2rem;padding:.1rem .5rem;border-radius:8px;background:rgba(255,255,255,.22)}
.theme-pick{background:rgba(255,255,255,.18)!important;color:#fff!important;border:1px solid rgba(255,255,255,.35)!important;border-radius:12px!important;
  font-weight:700;font-size:.8rem;padding:.3rem 2rem .3rem .7rem!important;width:auto;cursor:pointer}
.theme-pick option{color:#111827}
.theme-dots{display:flex;gap:4px;align-items:center;margin-right:8px}
.theme-dots span{width:10px;height:10px;border-radius:50%;background:#fff;opacity:.85}

.hero{position:relative;overflow:hidden;border-radius:26px;padding:28px 32px;margin-bottom:18px;color:#fff;
  box-shadow:0 22px 44px -20px var(--acc-glow);background-size:200% 200%!important;animation:gradShift 16s ease infinite}
.hero-a{background:var(--grad)} .hero-b{background:linear-gradient(120deg,#F97316,#EF4444 45%,#EC4899 75%,#8B5CF6)}
.hero-c{background:linear-gradient(120deg,#0EA5E9,#14B8A6 50%,#22C55E)} .hero-d{background:linear-gradient(120deg,#A855F7,var(--g2) 50%,#0EA5E9)}
.hero-e{background:linear-gradient(120deg,#DB2777,#7C3AED 50%,#2563EB)}
.hero h2{font-weight:800;letter-spacing:-.7px;margin:4px 0 6px;font-size:1.65rem}
.hero p{margin:0;opacity:.92;font-weight:500;max-width:760px}
.hero-badge{display:inline-block;font-size:.7rem;font-weight:800;letter-spacing:1px;text-transform:uppercase;padding:.2rem .6rem;border-radius:99px;background:rgba(255,255,255,.22)}
.hero-text{position:relative;z-index:2}
.blob{position:absolute;border-radius:50%;filter:blur(2px);opacity:.25;background:#fff;z-index:0}
.b1{width:220px;height:220px;right:-60px;top:-90px;animation:drift 9s ease-in-out infinite}
.b2{width:160px;height:160px;left:30%;bottom:-110px;animation:drift 12s ease-in-out infinite reverse}
@keyframes drift{50%{transform:translate(-20px,14px) scale(1.08)}}
.hero-net{position:absolute;right:24px;top:50%;transform:translateY(-50%);width:230px;opacity:.6;z-index:1}
.hero-net line{stroke:#fff;stroke-width:1.4;stroke-dasharray:6 5;animation:flow 2.2s linear infinite;opacity:.85}
.hero-net circle{fill:#fff;animation:glow 2.4s ease-in-out infinite}
.hero-net circle:nth-child(2n){animation-delay:.6s}.hero-net circle:nth-child(3n){animation-delay:1.2s}
@keyframes flow{to{stroke-dashoffset:-22}} @keyframes glow{0%,100%{opacity:.5}50%{opacity:1}}
@media (max-width:768px){.hero-net{display:none}}

.card{border-radius:var(--nl-radius)!important;border:1px solid rgba(148,163,184,.2)!important;box-shadow:var(--nl-shadow);
  transition:box-shadow .3s,transform .3s;overflow:hidden}
.card:hover{box-shadow:0 2px 4px rgba(15,23,42,.05),0 18px 44px -10px var(--acc-glow);transform:translateY(-2px)}
.card-header{background:transparent!important;font-weight:750;letter-spacing:-.2px;border-bottom:1px solid rgba(148,163,184,.16)!important;
  display:flex;align-items:center;gap:.6rem;padding:.8rem 1rem}
.ch-ic{width:32px;height:32px;border-radius:10px;display:grid;place-items:center;font-size:16px;flex:none;color:#fff;box-shadow:0 6px 14px -6px rgba(0,0,0,.35)}
.ci1{background:linear-gradient(135deg,#6366F1,#8B5CF6)} .ci2{background:linear-gradient(135deg,#14B8A6,#0EA5E9)}
.ci3{background:linear-gradient(135deg,#F59E0B,#F97316)} .ci4{background:linear-gradient(135deg,#EC4899,#F43F5E)}
.ci5{background:linear-gradient(135deg,#22C55E,#14B8A6)} .ci6{background:linear-gradient(135deg,#0EA5E9,#2563EB)}
.bslib-sidebar-layout>.sidebar{background:var(--bs-body-bg)!important;border-right:1px solid rgba(148,163,184,.2)!important}
.accordion-button{font-weight:750}
.accordion-button:not(.collapsed){background:var(--acc-soft);color:var(--g2)}
.accordion-item{border-color:rgba(148,163,184,.2)}
.form-check-input:checked{background-color:var(--g2);border-color:var(--g2)}
.form-control:focus,.form-select:focus,.selectize-input.focus{border-color:var(--g2)!important;box-shadow:0 0 0 .2rem var(--acc-soft)!important}
.irs--shiny .irs-bar{background:var(--grad);border:0}.irs--shiny .irs-single{background:var(--g2)}
.irs--shiny .irs-handle{border-color:var(--g2)}
.nav-underline .nav-link.active,.nav-underline .nav-link:hover{color:var(--g2)!important;border-bottom-color:var(--g2)!important}
.form-control,.form-select,.selectize-input{border-radius:12px!important}

.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:16px}
.kpi{display:flex;gap:14px;align-items:center;padding:16px 18px;border-radius:20px;color:#fff;position:relative;overflow:hidden;
  box-shadow:0 14px 30px -16px rgba(15,23,42,.55);transition:transform .25s,box-shadow .25s;animation:pop .5s cubic-bezier(.2,.8,.2,1.2) both}
.kpi:nth-child(2){animation-delay:.06s}.kpi:nth-child(3){animation-delay:.12s}.kpi:nth-child(4){animation-delay:.18s}.kpi:nth-child(5){animation-delay:.24s}
@keyframes pop{from{opacity:0;transform:translateY(12px) scale(.96)}}
.kpi:hover{transform:translateY(-4px) scale(1.01);box-shadow:0 20px 40px -16px rgba(15,23,42,.6)}
.kpi::after{content:'';position:absolute;right:-34px;top:-34px;width:120px;height:120px;border-radius:50%;background:rgba(255,255,255,.14)}
.kpi::before{content:'';position:absolute;left:-40%;top:0;width:40%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.25),transparent);transform:skewX(-20deg);transition:left .7s}
.kpi:hover::before{left:130%}
.kpi-icon{font-size:24px;width:48px;height:48px;border-radius:15px;display:grid;place-items:center;background:rgba(255,255,255,.22);flex:none}
.kpi-label{font-size:.7rem;text-transform:uppercase;letter-spacing:1px;font-weight:800;opacity:.88}
.kpi-value{display:block;font-size:1.6rem;font-weight:850;letter-spacing:-.6px;line-height:1.15;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi-sub{font-size:.75rem;opacity:.88;font-weight:550;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi-body{min-width:0;position:relative;z-index:1}
.kpi-small .kpi-value{font-size:1.1rem;line-height:1.35}
.k1{background:linear-gradient(135deg,#6366F1,#8B5CF6)} .k2{background:linear-gradient(135deg,#14B8A6,#0EA5E9)}
.k3{background:linear-gradient(135deg,#F59E0B,#F97316)} .k4{background:linear-gradient(135deg,#0EA5E9,#2563EB)}
.k5{background:linear-gradient(135deg,#EC4899,#F43F5E)} .k6{background:linear-gradient(135deg,#A855F7,#DB2777)}

.btn-glow{background:var(--grad);background-size:200% 200%;border:0;color:#fff!important;font-weight:800;border-radius:16px;padding:.8rem 1rem;width:100%;
  box-shadow:0 12px 26px -12px var(--acc-glow);transition:transform .2s,box-shadow .2s,background-position .6s;letter-spacing:-.1px;position:relative;overflow:hidden}
.btn-glow:hover{transform:translateY(-2px);background-position:100% 50%;box-shadow:0 18px 34px -12px var(--acc-glow)}
.btn-glow:active{transform:scale(.98)}
.btn-warm{background:linear-gradient(120deg,#F59E0B,#EF4444,#EC4899);background-size:200% 200%}
.btn-cool{background:linear-gradient(120deg,#0EA5E9,#14B8A6,#22C55E);background-size:200% 200%}
.btn-ai{background:linear-gradient(120deg,#DB2777,#7C3AED,#2563EB);background-size:200% 200%}
.btn-word{background:linear-gradient(120deg,#1D4ED8,#2563EB,#0EA5E9);background-size:200% 200%}
.btn-soft{border-radius:12px;font-weight:700}
.sticky-train{position:sticky;bottom:0;z-index:5;padding:12px 0 6px;background:linear-gradient(to top,var(--bs-body-bg) 75%,transparent)}

.chip{display:inline-flex;align-items:center;gap:.35rem;padding:.3rem .75rem;border-radius:999px;font-size:.78rem;font-weight:750;margin:2px}
.chip-v{background:rgba(99,102,241,.13);color:#4F46E5} .chip-t{background:rgba(20,184,166,.15);color:#0F8F80}
.chip-a{background:rgba(245,158,11,.16);color:#B45309} .chip-r{background:rgba(244,63,94,.13);color:#E11D48}
.chip-p{background:rgba(236,72,153,.13);color:#BE185D} .chip-g{background:rgba(100,116,139,.14);color:#64748B}
[data-bs-theme=dark] .chip-v{color:#A5B4FC}[data-bs-theme=dark] .chip-t{color:#5EEAD4}
[data-bs-theme=dark] .chip-a{color:#FCD34D}[data-bs-theme=dark] .chip-r{color:#FDA4AF}[data-bs-theme=dark] .chip-p{color:#F9A8D4}

.note{border-radius:14px;padding:.75rem 1rem;font-size:.86rem;line-height:1.55;border:1px solid transparent;margin:.4rem 0}
.note-i{background:rgba(14,165,233,.09);border-color:rgba(14,165,233,.25)} .note-w{background:rgba(245,158,11,.10);border-color:rgba(245,158,11,.3)}
.note-ok{background:rgba(34,197,94,.10);border-color:rgba(34,197,94,.3)} .note-e{background:rgba(244,63,94,.09);border-color:rgba(244,63,94,.3)}
.layer-row{border-left:4px solid var(--g2);padding:.25rem .7rem .1rem;border-radius:12px;background:var(--acc-soft);margin-bottom:.45rem}
.arch-strip{display:flex;align-items:center;flex-wrap:wrap;gap:6px;font-weight:750;font-size:.8rem;margin-top:.4rem}
.arch-strip .n{padding:.28rem .65rem;border-radius:10px;color:#fff;background:linear-gradient(135deg,#14B8A6,#0EA5E9)}
.arch-strip .n.in{background:linear-gradient(135deg,#6366F1,#8B5CF6)} .arch-strip .n.out{background:linear-gradient(135deg,#F59E0B,#F97316)}

pre.summary{font-family:'JetBrains Mono',Consolas,Menlo,monospace;font-size:.78rem;background:#0F172A;color:#C7D2FE;border-radius:14px;padding:14px;max-height:360px;overflow:auto;margin:0}
.empty-state{text-align:center;padding:60px 20px;color:var(--bs-secondary-color)}
.empty-state .big{font-size:60px;margin-bottom:10px;animation:float 3s ease-in-out infinite}
.pred-result{text-align:center;padding:20px;border-radius:20px;background:var(--grad);background-size:200% 200%;animation:gradShift 10s ease infinite;color:#fff;box-shadow:0 16px 30px -14px var(--acc-glow)}
.pred-result .v{font-size:2.4rem;font-weight:850;letter-spacing:-1px}
.prob-row{margin-bottom:.55rem}.prob-bar{height:12px;border-radius:12px;background:rgba(148,163,184,.18);overflow:hidden}
.prob-bar>div{height:100%;border-radius:12px;transition:width .6s cubic-bezier(.2,.8,.2,1)}

/* AI insights */
.ins{border-left:6px solid var(--st)!important}
.ins-good{--st:#0ca30c}.ins-warn{--st:#E8A200}.ins-bad{--st:#d03b3b}.ins-info{--st:#2a78d6}
.ins .card-body p{margin-bottom:.5rem;line-height:1.65}
.ins ul{margin:0;padding-left:1.1rem}.ins li{margin:.25rem 0;line-height:1.55}
.st-pill{margin-left:auto;font-size:.72rem;font-weight:800;padding:.22rem .65rem;border-radius:99px;color:#fff;background:var(--st)}
.ins-good .st-pill{color:#fff}.ins-warn .st-pill{color:#3b2a00}
.exec{position:relative;overflow:hidden;border-radius:24px;padding:24px 28px;color:#fff;background:var(--grad);background-size:200% 200%;
  animation:gradShift 14s ease infinite;box-shadow:0 18px 40px -18px var(--acc-glow);margin-bottom:16px}
.exec h4{font-weight:850;margin:0 0 8px}.exec p{font-size:1.02rem;line-height:1.7;margin:0;opacity:.96}
.ai-out{background:linear-gradient(135deg,rgba(219,39,119,.06),rgba(124,58,237,.06),rgba(37,99,235,.06));border-radius:16px;padding:14px 16px;margin-top:10px;
  border:1px solid rgba(124,58,237,.2)}
.ai-out .ai-h{font-weight:800;margin:.8rem 0 .3rem;background:linear-gradient(90deg,#DB2777,#7C3AED);-webkit-background-clip:text;background-clip:text;color:transparent}
.ai-out p,.ai-out li{line-height:1.6}

/* Training overlay */
#nl-overlay{position:fixed;inset:0;z-index:2000;display:none;place-items:center;background:rgba(15,23,42,.55);backdrop-filter:blur(8px)}
#nl-overlay.show{display:grid;animation:fadein .25s}
@keyframes fadein{from{opacity:0}}
.ov-card{width:min(440px,92vw);border-radius:28px;padding:28px 26px 22px;text-align:center;color:#fff;background:var(--grad);background-size:220% 220%;
  animation:gradShift 6s ease infinite;box-shadow:0 30px 70px -20px rgba(0,0,0,.6)}
.ov-net{width:100%;height:150px}
.ov-net line{stroke:rgba(255,255,255,.8);stroke-width:1.6;stroke-dasharray:5 6;animation:flow 1s linear infinite}
.ov-net circle{fill:#fff;transform-origin:center;transform-box:fill-box;animation:pulse 1.2s ease-in-out infinite}
.ov-net circle:nth-child(3n){animation-delay:.2s}.ov-net circle:nth-child(3n+1){animation-delay:.45s}
@keyframes pulse{50%{transform:scale(1.45);opacity:.7}}
.ov-card h3{font-weight:850;margin:.4rem 0 .2rem;letter-spacing:-.5px}
.ov-tip{min-height:2.6em;opacity:.92;font-weight:550;transition:opacity .3s}
.ov-time{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;margin:.4rem 0}
.ov-bar{height:6px;border-radius:6px;background:rgba(255,255,255,.25);overflow:hidden}
.ov-bar>div{height:100%;width:35%;border-radius:6px;background:#fff;animation:slide 1.3s ease-in-out infinite}
@keyframes slide{0%{transform:translateX(-110%)}100%{transform:translateX(320%)}}
.about-av{width:88px;height:88px;border-radius:28px;display:grid;place-items:center;font-size:42px;background:var(--grad);margin:0 auto 14px;box-shadow:0 16px 30px -10px var(--acc-glow);animation:float 4s ease-in-out infinite}
.shiny-notification{border-radius:16px;border:0;box-shadow:0 14px 34px rgba(15,23,42,.2);font-weight:600}
table.dataTable thead th{font-weight:750!important}
")

APP_JS <- "
(function(){
  function setTheme(v){ document.body.className = (document.body.className.replace(/\\bth-\\w+\\b/g,'') + ' th-' + v).trim(); }
  $(function(){ setTheme($('#ui_theme').val() || 'aurora'); });
  $(document).on('change', '#ui_theme', function(){ setTheme(this.value); });

  // Animated count-up for KPI numbers
  function countUp(el){
    if (el.dataset.done) return; el.dataset.done = 1;
    var t = parseFloat(el.dataset.count), d = parseInt(el.dataset.dec||0), s = el.dataset.suffix||'', p = el.dataset.prefix||'';
    if (!isFinite(t)) return;
    var t0 = performance.now(), dur = 950;
    function step(now){ var k = Math.min(1,(now-t0)/dur), e = 1-Math.pow(1-k,3);
      el.textContent = p + (t*e).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}) + s;
      if (k < 1) requestAnimationFrame(step); }
    requestAnimationFrame(step);
  }
  $(document).on('shiny:value', function(){ setTimeout(function(){ document.querySelectorAll('[data-count]').forEach(countUp); }, 60); });

  // Training overlay
  var timer = null, tipTimer = null;
  var tips = ['Initialising random weights…','Back-propagating errors…','Adjusting step sizes (rprop)…','Checking convergence threshold…',
              'Evaluating on held-out data…','Computing permutation importance…','Almost there…'];
  function showOverlay(title){
    $('#ov-title').text(title); $('#nl-overlay').addClass('show');
    var t0 = Date.now(), i = 0; $('#ov-tip').text(tips[0]);
    clearInterval(timer); clearInterval(tipTimer);
    timer = setInterval(function(){ $('#ov-time').text(((Date.now()-t0)/1000).toFixed(1) + ' s'); }, 100);
    tipTimer = setInterval(function(){ i = (i+1) % tips.length; $('#ov-tip').css('opacity',0); setTimeout(function(){ $('#ov-tip').text(tips[i]).css('opacity',.92); }, 250); }, 2400);
  }
  $(document).on('click', '#btn_train', function(){ showOverlay('Training neural network'); });
  $(document).on('click', '#btn_tune', function(){ showOverlay('Searching hyper-parameters'); });
  $(document).on('click', '#ai_go', function(){ showOverlay('Asking Claude for an interpretation'); });
  $(function(){ Shiny.addCustomMessageHandler('nl_done', function(x){ clearInterval(timer); clearInterval(tipTimer); $('#nl-overlay').removeClass('show'); }); });
})();
"

overlay_ui <- div(id = "nl-overlay", div(class = "ov-card",
                                         HTML('<svg class="ov-net" viewBox="0 0 300 150"><g>
    <line x1="30" y1="25" x2="150" y2="35"/><line x1="30" y1="25" x2="150" y2="75"/><line x1="30" y1="75" x2="150" y2="35"/><line x1="30" y1="75" x2="150" y2="115"/>
    <line x1="30" y1="125" x2="150" y2="75"/><line x1="30" y1="125" x2="150" y2="115"/><line x1="150" y1="35" x2="270" y2="55"/><line x1="150" y1="75" x2="270" y2="55"/>
    <line x1="150" y1="75" x2="270" y2="95"/><line x1="150" y1="115" x2="270" y2="95"/></g>
    <g><circle cx="30" cy="25" r="9"/><circle cx="30" cy="75" r="9"/><circle cx="30" cy="125" r="9"/><circle cx="150" cy="35" r="10"/>
    <circle cx="150" cy="75" r="10"/><circle cx="150" cy="115" r="10"/><circle cx="270" cy="55" r="11"/><circle cx="270" cy="95" r="11"/></g></svg>'),
                                         h3(id = "ov-title", "Training neural network"), div(class = "ov-time", id = "ov-time", "0.0 s"),
                                         div(class = "ov-tip", id = "ov-tip", ""), div(class = "ov-bar", div())))

theme <- bs_theme(
  version = 5, primary = "#6D28D9", secondary = "#64748B", success = "#14B8A6",
  info = "#0EA5E9", warning = "#F59E0B", danger = "#F43F5E",
  base_font = font_collection(font_link("Plus Jakarta Sans",
                                        href = "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap"),
                              "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"),
  code_font = font_collection(font_link("JetBrains Mono",
                                        href = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap"), "Consolas", "Menlo", "monospace"),
  "border-radius" = "0.8rem", "font-size-base" = "0.92rem"
)

demo_choices <- c(setNames(names(DEMOS), sapply(DEMOS, function(d)
  paste0(if (d$type == "classification") "🔵 " else "🟢 ", d$label))), "📁 Upload my own file" = "upload")

explorer_types <- c("✨ Scatter" = "sc", "🧊 3-D scatter" = "s3", "🔢 Scatter matrix" = "sm", "🎻 Violin" = "vi",
                    "🌊 Density" = "de", "📊 Histogram" = "hi", "🔥 Correlation" = "co", "🧵 Parallel coords" = "pc")

ui <- page_navbar(
  id = "nav", theme = theme, fillable = FALSE,
  title = div(class = "brand", div(class = "brand-logo", "🧠"), "NeuroLab", tags$small("v8")),
  header = tagList(tags$head(tags$style(HTML(APP_CSS)), tags$script(HTML(APP_JS))), overlay_ui),
  
  # ════════════════════ BUILD ════════════════════
  nav_panel("Build & Train", value = "build", icon = span("🏗️"),
            layout_sidebar(fillable = FALSE,
                           sidebar = sidebar(width = 360, open = "desktop",
                                             accordion(id = "acc", multiple = TRUE, open = c("Data", "Architecture", "Training"),
                                                       accordion_panel("Data", icon = span("📦"),
                                                                       selectInput("data_src", "Dataset", choices = demo_choices, selected = "iris"),
                                                                       uiOutput("demo_desc"),
                                                                       conditionalPanel("input.data_src == 'upload'",
                                                                                        fileInput("csv_file", NULL, accept = c(".csv", ".tsv", ".txt"), buttonLabel = "Browse…", placeholder = "CSV / TSV / ; separated")),
                                                                       uiOutput("target_ui"),
                                                                       radioButtons("ptype", "Problem type", inline = TRUE,
                                                                                    choices = c("Auto" = "auto", "Classification" = "classification", "Regression" = "regression")),
                                                                       uiOutput("type_badge"),
                                                                       uiOutput("exclude_ui"),
                                                                       sliderInput("split_pct", "Training set (%)", 50, 90, 75, 5),
                                                                       selectInput("scaling", "Feature scaling", c("Min–max [0, 1]" = "minmax", "Z-score" = "zscore", "None" = "none"))
                                                       ),
                                                       accordion_panel("Architecture", icon = span("🔷"),
                                                                       numericInput("n_layers", "Hidden layers", 2, 1, 5),
                                                                       uiOutput("layer_sliders"),
                                                                       selectInput("act", "Activation function", c("Logistic (sigmoid)" = "logistic", "Tanh" = "tanh", "Softplus (smooth ReLU)" = "softplus")),
                                                                       uiOutput("arch_strip")
                                                       ),
                                                       accordion_panel("Training", icon = span("🎛️"),
                                                                       selectInput("algo", "Algorithm", c("rprop+ (fast, default)" = "rprop+", "rprop−" = "rprop-", "sag" = "sag", "slr" = "slr", "backprop (uses LR)" = "backprop")),
                                                                       conditionalPanel("input.algo == 'backprop'", numericInput("lr", "Learning rate", 0.01, 0.0001, 1, 0.001)),
                                                                       selectInput("err", "Error function", c("Auto (recommended)" = "auto", "Sum of squared errors" = "sse", "Cross-entropy" = "ce")),
                                                                       layout_columns(col_widths = c(6, 6),
                                                                                      selectInput("thresh", "Stop threshold", c("Auto" = "auto", "0.1", "0.05", "0.01", "0.005", "0.001"), "auto"),
                                                                                      numericInput("stepmax", "Max steps", 50000, 1000, 1e6, 10000)),
                                                                       layout_columns(col_widths = c(6, 6),
                                                                                      numericInput("reps", "Repetitions", 1, 1, 10),
                                                                                      numericInput("seed", "Random seed", 2024, 1, 99999)),
                                                                       checkboxInput("auto_relax", "Auto-relax threshold if it doesn't converge", TRUE)
                                                       )
                                             ),
                                             div(class = "sticky-train", actionButton("btn_train", "🚀  Train network", class = "btn-glow"), uiOutput("train_status"))
                           ),
                           hero("Neural Network Model Builder", "Load a dataset, shape the architecture and train. Classification and regression, numeric and categorical features — all handled.", badge = "Step 1 · Build"),
                           uiOutput("data_kpis"),
                           navset_card_underline(
                             nav_panel("📋 Data", spin(DTOutput("tbl_data"))),
                             nav_panel("🧪 Column profile", spin(DTOutput("tbl_profile"))),
                             nav_panel("📊 Explorer",
                                       layout_columns(col_widths = c(3, 2, 2, 2, 3),
                                                      selectInput("plt_type", "Chart", explorer_types),
                                                      uiOutput("sel_x"), uiOutput("sel_y"), uiOutput("sel_z"), uiOutput("sel_col")),
                                       spin(plotlyOutput("plt_feat", height = "480px"))),
                             nav_panel("📖 Guide",
                                       div(class = "p-2",
                                           h5("Choosing settings", class = "fw-bold"),
                                           tags$table(class = "table table-sm align-middle",
                                                      tags$thead(tags$tr(tags$th("Setting"), tags$th("Classification"), tags$th("Regression"))),
                                                      tags$tbody(
                                                        tags$tr(tags$td("Output neurons"), tags$td("1 (binary) or 1 per class"), tags$td("1 (linear)")),
                                                        tags$tr(tags$td("Activation"), tags$td("Logistic or Tanh"), tags$td("Logistic / Tanh / Softplus")),
                                                        tags$tr(tags$td("Error function"), tags$td("Cross-entropy (logistic) or SSE"), tags$td("SSE")),
                                                        tags$tr(tags$td("Hidden layers"), tags$td("1–2"), tags$td("1–2")),
                                                        tags$tr(tags$td("Neurons / layer"), tags$td("3–10"), tags$td("3–10")))),
                                           div(class = "note note-i", tags$b("Why does training sometimes stop early? "), "neuralnet stops when all partial derivatives of the summed error fall below the threshold. 'Auto' scales it with rows and outputs; auto-relax raises it further if Max steps is reached."),
                                           div(class = "note note-i", tags$b("Learning rate "), "is only used by the classic ", tags$code("backprop"), " algorithm; it is lowered automatically if training diverges."),
                                           div(class = "note note-i", tags$b("Regression outputs "), "are bounded to the training target range ± 25 % so unbounded activations can't extrapolate to absurd values."),
                                           div(class = "note note-i", tags$b("Categorical predictors "), "are one-hot encoded; columns with more than 20 levels (IDs, names) are dropped.")))
                           )
            )
  ),
  
  nav_panel("Results", value = "results", icon = span("📈"),
            hero("Results & Diagnostics", "Held-out test performance, train-vs-test comparison and prediction diagnostics.", "hero-c", badge = "Step 2 · Evaluate"),
            uiOutput("results_ui")),
  
  nav_panel("Interpretation", value = "interp", icon = span("🌐"),
            hero("Inside the Network", "Architecture, weight flow, partial dependence and what drives the predictions.", "hero-d", badge = "Step 3 · Understand"),
            uiOutput("interp_ui")),
  
  nav_panel("AI Insights & Report", value = "ai", icon = span("✨"),
            hero("AI Insights & Word Report", "A statistician-style interpretation of your model, plus a one-click Word report with tables and figures.", "hero-e", badge = "Step 4 · Report"),
            uiOutput("ai_ui")),
  
  nav_panel("Tuning", value = "tune", icon = span("🎯"),
            hero("Hyperparameter Search", "Grid search over architecture and activation, with holdout or k-fold cross-validation.", "hero-b", badge = "Optimise"),
            layout_columns(col_widths = c(4, 8),
                           card(ch("🔎", "Search grid", 3),
                                textInput("t_neurons", "Neurons per layer (comma-separated)", "3,6,10"),
                                checkboxGroupInput("t_layers", "Hidden layers", c("1" = 1, "2" = 2, "3" = 3), selected = c(1, 2), inline = TRUE),
                                checkboxGroupInput("t_act", "Activations", c("Logistic" = "logistic", "Tanh" = "tanh", "Softplus" = "softplus"), selected = c("logistic", "tanh"), inline = TRUE),
                                conditionalPanel("input.algo == 'backprop'", textInput("t_lr", "Learning rates (backprop only)", "0.005,0.01,0.05")),
                                layout_columns(col_widths = c(6, 6),
                                               selectInput("t_folds", "Validation", c("Holdout 75/25" = 1, "3-fold CV" = 3, "5-fold CV" = 5)),
                                               selectInput("t_thresh", "Threshold", c("Auto" = "auto", "0.1", "0.05", "0.01"), "auto")),
                                numericInput("t_steps", "Max steps / fit", 30000, 1000, 5e5, 5000),
                                uiOutput("tune_size"),
                                actionButton("btn_tune", "🔬  Run grid search", class = "btn-glow btn-warm"),
                                uiOutput("tune_best"), uiOutput("tune_apply_ui")),
                           navset_card_underline(
                             nav_panel("🗺️ Heatmap", spin(plotlyOutput("plt_heat", height = "460px"))),
                             nav_panel("🫧 Landscape", spin(plotlyOutput("plt_land", height = "460px"))),
                             nav_panel("🏁 Ranking", spin(plotlyOutput("plt_rank", height = "460px"))),
                             nav_panel("📋 Table", spin(DTOutput("tbl_tune")), downloadButton("dl_tune", "Download CSV", class = "btn-soft btn-outline-primary mt-2"))))),
  
  nav_panel("Predict", value = "predict", icon = span("🔮"),
            hero("Score New Data", "Try a single what-if case interactively, or upload a file to score in bulk.", badge = "Deploy"),
            uiOutput("predict_ui")),
  
  nav_panel("About", value = "about", icon = span("ℹ️"),
            layout_columns(col_widths = c(6, 6),
                           card(card_body(class = "text-center",
                                          div(class = "about-av", "🧠"), h3("NeuroLab v8", class = "fw-bold"),
                                          p(class = "text-secondary", "An interactive studio for building, tuning, interpreting and reporting feed-forward neural networks in R — no coding required."),
                                          hr(),
                                          div(class = "text-start", tags$ul(class = "lh-lg",
                                                                            tags$li("Classification (binary & multiclass) and regression"),
                                                                            tags$li("Numeric + categorical predictors, missing-value imputation"),
                                                                            tags$li("Logistic, Tanh, Softplus activations · 5 training algorithms"),
                                                                            tags$li("Auto-recovery when training does not converge"),
                                                                            tags$li("Network diagram, Sankey weight flow, PDP / ICE, permutation & Olden importance"),
                                                                            tags$li("AI insight engine + optional Claude narrative"),
                                                                            tags$li("One-click Word report with figures and interpretation"),
                                                                            tags$li("Grid search with k-fold CV · what-if predictor · batch scoring"))),
                                          div(lapply(c("neuralnet", "shiny", "bslib", "plotly", "ggplot2", "officer", "flextable", "DT"), function(p) span(class = "chip chip-v", p))))),
                           card(card_body(class = "text-center",
                                          div(class = "about-av", "👨‍💻"), h5("Developer", class = "text-secondary"),
                                          h3("M. Iqbal Jeelani, Arif Bashir,Imran Khan, Immad A.Shah, Nageena Nazir, Mansha Gul", class = "fw-bold"), p("SKUAST-Kashmir · Scientist · Statistician · R Programmer"),
                                          tags$a(class = "btn btn-outline-danger btn-soft m-1", href = "mailto:jeelani.mqbal@gmail.com", "✉️ jeelani.miqbal@gmail.com"),
                                          tags$a(class = "btn btn-outline-warning btn-soft m-1", href = "https://youtube.com/@Iqbalstat", target = "_blank", "▶️ YouTube"),
                                          hr(), h6("Demo datasets", class = "fw-bold text-start"),
                                          tags$table(class = "table table-sm text-start", tags$tbody(lapply(DEMOS, function(d) tags$tr(tags$td(d$label),
                                                                                                                                       tags$td(span(class = paste("chip", if (d$type == "classification") "chip-v" else "chip-t"), d$type)),
                                                                                                                                       tags$td(class = "text-secondary", d$desc))))))))),
  
  nav_spacer(),
  nav_item(div(class = "d-flex align-items-center",
               div(class = "theme-dots", span(), span(), span()),
               tags$select(id = "ui_theme", class = "form-select form-select-sm theme-pick", title = "Colour theme",
                           lapply(names(THEMES), function(k) tags$option(value = k, paste("🎨", THEMES[[k]]$name)))))),
  nav_item(input_dark_mode(id = "dark", mode = "light"))
)

# ─────────────────────────────────────────────────────────────────────────────
#  SERVER
# ─────────────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {
  
  notify <- function(msg, type = "message", dur = 6) showNotification(HTML(msg), type = type, duration = dur)
  done_overlay <- function() session$sendCustomMessage("nl_done", list())
  rv <- reactiveValues(fit = NULL, ins = NULL, ai = NULL, tune = NULL, score = NULL, pending_hidden = NULL, train_msg = NULL)
  TH  <- reactive(THEMES[[input$ui_theme %||% "aurora"]] %||% THEMES$aurora)
  HUE <- reactive(TH()$seq)
  
  # ── Data source ──────────────────────────────────────────────────────────
  uploaded <- reactive({
    req(input$csv_file)
    tryCatch(read_any(input$csv_file$datapath),
             error = function(e) { notify(paste("<b>Could not read file:</b>", e$message), "error"); NULL })
  })
  
  dataset <- reactive({
    src <- input$data_src; req(src)
    if (src == "upload") {
      d <- uploaded(); req(d)
      return(list(df = d, name = input$csv_file$name, target = names(d)[ncol(d)], type = "auto"))
    }
    d <- tryCatch(load_demo(src), error = function(e) e)
    validate(need(is.data.frame(d), if (inherits(d, "error")) conditionMessage(d) else "Dataset could not be loaded."))
    dm <- DEMOS[[src]]
    list(df = d, name = dm$label, target = dm$target, type = dm$type)
  })
  
  observeEvent(input$data_src, {
    updateRadioButtons(session, "ptype", selected = if (input$data_src == "upload") "auto" else DEMOS[[input$data_src]]$type)
  })
  
  output$demo_desc <- renderUI({
    if (input$data_src == "upload") return(div(class = "note note-i", "Upload a CSV/TSV with a header row, then pick the target column."))
    d <- tryCatch(dataset()$df, error = function(e) NULL)
    div(class = "text-secondary small mb-2", DEMOS[[input$data_src]]$desc, if (!is.null(d)) paste0(" · ", nrow(d), " rows loaded"))
  })
  
  output$target_ui <- renderUI({
    ds <- dataset(); req(ds)
    selectInput("target", "Target column", choices = names(ds$df), selected = ds$target)
  })
  
  output$exclude_ui <- renderUI({
    ds <- dataset(); req(ds, input$target)
    selectizeInput("exclude", "Exclude predictors (optional)", choices = setdiff(names(ds$df), input$target),
                   multiple = TRUE, options = list(placeholder = "e.g. ID columns"))
  })
  
  resolved_type <- reactive({
    ds <- dataset(); req(ds, input$target, input$target %in% names(ds$df))
    if (input$ptype == "auto") detect_type(ds$df[[input$target]]) else input$ptype
  })
  
  output$type_badge <- renderUI({
    tp <- resolved_type(); ds <- dataset(); y <- ds$df[[input$target]]
    info <- if (tp == "classification") paste(length(unique(na.omit(y))), "classes") else
      paste0("range ", signif(min(to_num(y), na.rm = TRUE), 4), " – ", signif(max(to_num(y), na.rm = TRUE), 4))
    warn <- if (tp == "regression" && num_share(y) < .95)
      div(class = "note note-e", "This target is not numeric — regression won't work. Choose Classification.")
    else if (tp == "classification" && length(unique(na.omit(y))) > 30)
      div(class = "note note-w", "Many unique values — this looks like a regression target.")
    tagList(div(class = "mb-2",
                span(class = paste("chip", if (tp == "classification") "chip-v" else "chip-t"), if (tp == "classification") "🔵 Classification" else "🟢 Regression"),
                span(class = "chip chip-g", info)), warn)
  })
  
  output$data_kpis <- renderUI({
    ds <- dataset(); req(ds, input$target)
    d <- ds$df; pr <- setdiff(names(d), c(input$target, input$exclude))
    nnum <- sum(vapply(d[pr], function(v) is.numeric(v) || num_share(v) >= .95, TRUE))
    tp <- tryCatch(resolved_type(), error = function(e) "—")
    div(class = "kpi-row",
        kpi("Dataset", ds$name, "🗂️", "k1", small = TRUE),
        kpi("Rows", format(nrow(d), big.mark = ","), "🧾", "k4", num = nrow(d)),
        kpi("Predictors", length(pr), "🧩", "k2", sub = paste(nnum, "numeric ·", length(pr) - nnum, "categorical"), num = length(pr)),
        kpi("Missing cells", sum(is.na(d)), "🕳️", "k3", num = sum(is.na(d))),
        kpi("Task", tools::toTitleCase(tp), "🎯", "k6", sub = paste("target:", input$target), small = TRUE))
  })
  
  output$tbl_data <- renderDT({
    ds <- dataset(); req(ds)
    datatable(ds$df, rownames = FALSE, class = "compact hover stripe", options = list(scrollX = TRUE, pageLength = 10, dom = "ftip"))
  })
  
  output$tbl_profile <- renderDT({
    ds <- dataset(); req(ds, input$target); d <- ds$df
    isn <- vapply(d, function(v) is.numeric(v) || num_share(v) >= .95, TRUE)
    prof <- data.frame(
      Column = names(d),
      Role = ifelse(names(d) == input$target, "🎯 target", ifelse(names(d) %in% input$exclude, "⛔ excluded", "predictor")),
      Type = ifelse(isn, "numeric", "categorical"),
      Missing = vapply(d, function(v) sum(is.na(v)), 1L),
      Unique = vapply(d, function(v) length(unique(na.omit(v))), 1L),
      Summary = vapply(seq_along(d), function(j) { v <- d[[j]]
      if (isn[j]) { x <- to_num(v); sprintf("min %s · median %s · max %s", signif(min(x, na.rm = TRUE), 4), signif(stats::median(x, na.rm = TRUE), 4), signif(max(x, na.rm = TRUE), 4)) }
      else { t <- sort(table(v), decreasing = TRUE); paste(head(paste0(names(t), " (", t, ")"), 4), collapse = ", ") } }, ""),
      stringsAsFactors = FALSE)
    datatable(prof, rownames = FALSE, class = "compact hover", options = list(dom = "t", pageLength = 100, scrollX = TRUE)) |>
      formatStyle("Missing", background = styleColorBar(c(0, max(1, prof$Missing)), alpha_col(SERIES[2], .35)), backgroundSize = "95% 70%", backgroundRepeat = "no-repeat", backgroundPosition = "center") |>
      formatStyle("Type", color = styleEqual(c("numeric", "categorical"), c(SERIES[1], SERIES[2])), fontWeight = "bold")
  })
  
  # ── Explorer ──────────────────────────────────────────────────────────────
  num_vars <- reactive({ d <- dataset()$df; names(d)[vapply(d, function(v) is.numeric(v) || num_share(v) >= .95, TRUE)] })
  output$sel_x <- renderUI(selectInput("fx", "X", num_vars(), num_vars()[1]))
  output$sel_y <- renderUI(selectInput("fy", "Y", num_vars(), num_vars()[min(2, length(num_vars()))]))
  output$sel_z <- renderUI(selectInput("fz", "Z (3-D)", num_vars(), num_vars()[min(3, length(num_vars()))]))
  output$sel_col <- renderUI(selectInput("fc", "Colour / group", c("None", names(dataset()$df)), input$target))
  
  output$plt_feat <- renderPlotly({
    d <- dataset()$df; req(input$fx, input$plt_type); typ <- input$plt_type; hue <- HUE()
    nv <- num_vars()
    grp <- if (!is.null(input$fc) && input$fc != "None" && input$fc %in% names(d)) d[[input$fc]] else NULL
    grp_num <- !is.null(grp) && (is.numeric(grp) || num_share(grp) >= .95) && length(unique(grp)) > 10
    gf <- if (!is.null(grp) && !grp_num) factor(ifelse(is.na(grp), "NA", as.character(grp))) else NULL
    gcols <- if (!is.null(gf)) level_cols(levels(gf)) else NULL
    x <- to_num(d[[input$fx]])
    
    if (typ == "co") {
      if (length(nv) < 2) return(empty_plot("Need ≥ 2 numeric columns"))
      M <- sapply(d[nv], to_num); cm <- round(stats::cor(M, use = "pairwise.complete.obs"), 2)
      return(plot_ly(x = nv, y = nv, z = cm, type = "heatmap", zmin = -1, zmax = 1, colorscale = div_scale, xgap = 3, ygap = 3,
                     text = cm, texttemplate = "%{text}", hovertemplate = "%{x} × %{y}: r = %{z}<extra></extra>",
                     colorbar = list(title = "r", thickness = 12)) |>
               pl(legend = FALSE) |> layout(yaxis = list(autorange = "reversed")))
    }
    if (typ == "pc") {
      vv <- head(nv, 12); dims <- lapply(vv, function(v) list(label = v, values = to_num(d[[v]])))
      line <- if (!is.null(gf)) {
        k <- nlevels(gf); cols <- unname(gcols)
        cs <- if (k == 1) list(list(0, cols[1]), list(1, cols[1])) else unlist(lapply(seq_len(k), function(i) list(list((i - 1) / k, cols[i]), list(i / k - 1e-6, cols[i]))), recursive = FALSE)
        list(color = as.integer(gf), colorscale = cs, cmin = .5, cmax = k + .5, showscale = TRUE,
             colorbar = list(tickvals = seq_len(k), ticktext = levels(gf), thickness = 12, title = input$fc))
      } else if (grp_num) list(color = to_num(grp), colorscale = seq_scale(hue), showscale = TRUE, colorbar = list(title = input$fc, thickness = 12))
      else list(color = hue)
      return(plot_ly(type = "parcoords", line = line, dimensions = dims, labelfont = list(size = 12), tickfont = list(size = 10)) |> pl(legend = FALSE))
    }
    if (typ == "sm") {
      vv <- head(nv, 6); dims <- lapply(vv, function(v) list(label = v, values = to_num(d[[v]])))
      if (!is.null(gf)) {
        p <- plot_ly()
        for (lv in levels(gf)) { k <- gf == lv
        p <- add_trace(p, type = "splom", name = lv, dimensions = lapply(vv, function(v) list(label = v, values = to_num(d[[v]])[k])),
                       marker = list(color = gcols[[lv]], size = 6, opacity = .75, line = list(color = "white", width = .6)), diagonal = list(visible = FALSE)) }
      } else p <- plot_ly(type = "splom", dimensions = dims, marker = list(color = hue, size = 6, opacity = .7, line = list(color = "white", width = .6)), diagonal = list(visible = FALSE))
      return(p |> pl() |> layout(dragmode = "select"))
    }
    if (typ == "s3") {
      req(input$fy, input$fz); y <- to_num(d[[input$fy]]); z <- to_num(d[[input$fz]])
      p <- if (!is.null(gf)) plot_ly(x = x, y = y, z = z, color = gf, colors = unname(gcols), type = "scatter3d", mode = "markers",
                                     marker = list(size = 5, opacity = .85, line = list(color = "white", width = .5)))
      else plot_ly(x = x, y = y, z = z, type = "scatter3d", mode = "markers",
                   marker = list(size = 5, color = if (grp_num) to_num(grp) else z, colorscale = seq_scale(hue), showscale = TRUE, opacity = .9, colorbar = list(thickness = 12)))
      return(p |> pl() |> layout(scene = list(xaxis = list(title = input$fx), yaxis = list(title = input$fy), zaxis = list(title = input$fz))))
    }
    if (typ == "sc") {
      req(input$fy); y <- to_num(d[[input$fy]])
      p <- if (!is.null(gf)) plot_ly(x = x, y = y, color = gf, colors = unname(gcols), type = "scatter", mode = "markers",
                                     marker = list(size = 10, opacity = .8, line = list(color = "white", width = 1.2)))
      else plot_ly(x = x, y = y, type = "scatter", mode = "markers",
                   marker = list(size = 10, opacity = .85, color = if (grp_num) to_num(grp) else y, colorscale = seq_scale(hue),
                                 showscale = grp_num, line = list(color = "white", width = 1.2), colorbar = list(title = input$fc, thickness = 12)))
      ok <- is.finite(x) & is.finite(y)
      if (sum(ok) > 10) {
        lo <- tryCatch(stats::loess(y ~ x, data = data.frame(x = x[ok], y = y[ok])), error = function(e) NULL)
        if (!is.null(lo)) { xs <- seq(min(x[ok]), max(x[ok]), length.out = 80)
        p <- add_trace(p, x = xs, y = predict(lo, data.frame(x = xs)), type = "scatter", mode = "lines", name = "Trend (loess)",
                       line = list(color = "#334155", width = 2.5, dash = "dot"), inherit = FALSE, marker = NULL) }
      }
      return(pl(p, input$fx, input$fy))
    }
    if (typ == "vi") {
      g <- if (is.null(gf)) factor(rep("All", length(x))) else gf; cols <- level_cols(levels(g))
      p <- plot_ly()
      for (lv in levels(g)) p <- add_trace(p, x = g[g == lv], y = x[g == lv], type = "violin", name = lv, box = list(visible = TRUE),
                                           meanline = list(visible = TRUE), points = "all", jitter = .4, pointpos = 0,
                                           fillcolor = alpha_col(cols[[lv]], .35), line = list(color = cols[[lv]], width = 2),
                                           marker = list(color = cols[[lv]], size = 4, opacity = .55))
      return(pl(p, input$fc, input$fx, legend = FALSE))
    }
    if (typ == "de") {
      g <- if (is.null(gf)) factor(rep("All", length(x))) else gf; cols <- level_cols(levels(g)); p <- plot_ly()
      for (lv in levels(g)) { xx <- x[g == lv & is.finite(x)]; if (length(xx) < 3) next
      dd <- stats::density(xx)
      p <- add_trace(p, x = dd$x, y = dd$y, type = "scatter", mode = "lines", fill = "tozeroy", name = lv,
                     line = list(color = if (nlevels(g) == 1) hue else cols[[lv]], width = 3, shape = "spline"),
                     fillcolor = alpha_col(if (nlevels(g) == 1) hue else cols[[lv]], .22)) }
      return(pl(p, input$fx, "Density"))
    }
    p <- if (!is.null(gf)) plot_ly(x = x, color = gf, colors = unname(gcols), type = "histogram", opacity = .72, marker = list(line = list(color = "white", width = 1))) |> layout(barmode = "overlay")
    else plot_ly(x = x, type = "histogram", marker = list(color = hue, line = list(color = "white", width = 1.5)))
    pl(p, input$fx, "Count")
  })
  
  # ── Architecture inputs ───────────────────────────────────────────────────
  output$layer_sliders <- renderUI({
    n <- max(1, min(5, input$n_layers %||% 1))
    defaults <- rv$pending_hidden %||% c(5, 3, 3, 3, 3)
    tagList(lapply(seq_len(n), function(i) div(class = "layer-row",
                                               sliderInput(paste0("neu_", i), paste0("Layer ", i, " neurons"), 1, 64,
                                                           value = isolate(input[[paste0("neu_", i)]]) %||% defaults[min(i, length(defaults))], step = 1))))
  })
  observeEvent(rv$pending_hidden, {
    h <- rv$pending_hidden; for (i in seq_along(h)) updateSliderInput(session, paste0("neu_", i), value = h[i])
  }, ignoreInit = TRUE)
  
  hidden_vec <- reactive({
    n <- max(1, min(5, input$n_layers %||% 1))
    vapply(seq_len(n), function(i) as.integer(input[[paste0("neu_", i)]] %||% 4L), 1L)
  })
  
  output$arch_strip <- renderUI({
    h <- hidden_vec(); tp <- tryCatch(resolved_type(), error = function(e) "regression")
    ds <- tryCatch(dataset(), error = function(e) NULL); req(ds)
    nout <- if (tp == "regression") 1 else { k <- length(unique(na.omit(ds$df[[input$target]]))); if (k == 2) 1 else k }
    div(class = "arch-strip", span(class = "n in", "Inputs"), "→", lapply(h, function(x) tagList(span(class = "n", x), "→")), span(class = "n out", paste(nout, "out")))
  })
  
  # ── TRAIN ─────────────────────────────────────────────────────────────────
  observeEvent(input$btn_train, {
    on.exit(done_overlay(), add = TRUE)
    ds <- tryCatch(dataset(), error = function(e) NULL)
    if (is.null(ds) || is.null(input$target)) { notify("Load a dataset first.", "warning"); return() }
    rv$train_msg <- NULL
    withProgress(message = "Preparing data…", value = .1, {
      prep <- tryCatch(nl_prepare(ds$df, input$target, input$ptype, input$exclude %||% character(), input$split_pct / 100, input$scaling, input$seed),
                       error = function(e) { rv$train_msg <- list(type = "e", text = e$message); NULL })
      if (is.null(prep)) { notify(paste("<b>Data problem:</b>", rv$train_msg$text), "error"); return() }
      if (nrow(prep$Xtr) > 3000) notify("Large training set — neuralnet may take a while.", "warning")
      setProgress(.3, message = sprintf("Training %s network…", paste(c(ncol(prep$Xtr), hidden_vec(), ncol(prep$Ytr)), collapse = "-")))
      tr <- tryCatch(nl_train(prep, hidden_vec(), input$act, input$algo, input$lr %||% .01, input$thresh, input$stepmax, input$reps,
                              input$err, input$seed, isTRUE(input$auto_relax)),
                     error = function(e) { rv$train_msg <- list(type = "e", text = e$message); NULL })
      if (is.null(tr)) { notify(paste("<b>Training failed.</b>", rv$train_msg$text), "error", 10); return() }
      setProgress(.8, message = "Evaluating & interpreting…")
      fit <- tryCatch(nl_evaluate(prep, tr), error = function(e) { rv$train_msg <- list(type = "e", text = e$message); NULL })
      if (is.null(fit)) { notify(paste("<b>Evaluation failed:</b>", rv$train_msg$text), "error"); return() }
      fit$dataset <- ds$name
      rv$fit <- fit; rv$score <- NULL; rv$ai <- NULL
      rv$ins <- tryCatch(nl_insights(fit), error = function(e) { notify(paste("Insight engine:", e$message), "warning"); NULL })
      relaxed <- tr$threshold > tr$threshold_req
      rv$train_msg <- list(type = if (relaxed) "w" else "ok",
                           text = paste0("✅ Trained in ", round(tr$secs, 1), "s · ", format(tr$steps, big.mark = ","), " steps",
                                         if (relaxed) paste0(" · threshold relaxed to ", tr$threshold) else ""))
    })
    notify("✅ <b>Model trained!</b> Opening results…", "message", 3)
    nav_select("nav", "results")
  })
  
  output$train_status <- renderUI({ m <- rv$train_msg; if (is.null(m)) return(NULL); div(class = paste0("note mt-2 note-", m$type), m$text) })
  
  no_model <- function() div(class = "empty-state", div(class = "big", "🧠"),
                             h4("No model trained yet"), p("Configure and train a network on the Build & Train tab."),
                             actionButton("go_build", "Go to Build & Train", class = "btn-glow", style = "width:auto"))
  observeEvent(input$go_build, nav_select("nav", "build"))
  
  # ── RESULTS ───────────────────────────────────────────────────────────────
  output$results_ui <- renderUI({
    fit <- rv$fit; if (is.null(fit)) return(no_model())
    tr <- fit$train; m <- fit$test$metrics; reg <- fit$rec$type == "regression"
    k <- if (reg) div(class = "kpi-row",
                      kpi("R² (test)", f3(m[["R2"]]), "📐", "k1", num = m[["R2"]], dec = 3, sub = paste(grade_r2(m[["R2"]]), "fit")),
                      kpi("RMSE", signif(m[["RMSE"]], 4), "📏", "k2", num = m[["RMSE"]], dec = 3),
                      kpi("MAE", signif(m[["MAE"]], 4), "📉", "k4", num = m[["MAE"]], dec = 3),
                      kpi("Pearson r", f3(m[["r"]]), "🔗", "k6", num = m[["r"]], dec = 3),
                      kpi("MAPE", if (is.na(m[["MAPE"]])) "—" else sprintf("%.1f%%", m[["MAPE"]]), "％", "k3", num = if (is.na(m[["MAPE"]])) NULL else m[["MAPE"]], dec = 1, suffix = "%"))
    else { o <- m$overall; div(class = "kpi-row",
                               kpi("Accuracy (test)", pc(o[["Accuracy"]]), "🎯", "k1", num = 100 * o[["Accuracy"]], dec = 1, suffix = "%"),
                               kpi("Cohen's κ", f3(o[["Kappa"]]), "🤝", "k2", num = o[["Kappa"]], dec = 3, sub = grade_kappa(o[["Kappa"]])),
                               kpi("Balanced acc.", pc(o[["BalAcc"]]), "⚖️", "k4", num = 100 * o[["BalAcc"]], dec = 1, suffix = "%"),
                               kpi("Macro F1", f3(o[["MacroF1"]]), "🧮", "k6", num = o[["MacroF1"]], dec = 3),
                               if (!is.na(o[["AUC"]])) kpi("ROC AUC", f3(o[["AUC"]]), "📈", "k3", num = o[["AUC"]], dec = 3, sub = paste(grade_auc(o[["AUC"]]), "discrimination"))) }
    notes <- c(fit$msgs, tr$notes)
    tagList(k,
            div(class = "mb-3",
                span(class = "chip chip-v", paste("🧠", paste(c(length(fit$rec$features), tr$hidden, ncol(tail(tr$model$weights[[tr$best]], 1)[[1]])), collapse = " → "))),
                span(class = "chip chip-t", ACT_LABELS[[tr$act]]), span(class = "chip chip-a", tr$algorithm),
                span(class = "chip chip-p", paste("error fn:", tr$err)),
                span(class = "chip chip-g", paste0(fit$n["train"], " train / ", fit$n["test"], " test")),
                span(class = "chip chip-g", paste0("⏱ ", round(tr$secs, 1), "s")),
                actionLink("go_ai", span(class = "chip chip-r", "✨ See AI interpretation →"))),
            if (length(notes)) div(class = "note note-w", tags$b("Notes: "), HTML(paste(htmltools::htmlEscape(notes), collapse = "<br>"))),
            layout_columns(col_widths = c(7, 5),
                           card(full_screen = TRUE, ch(if (reg) "🎯" else "🧩", if (reg) "Actual vs predicted (test)" else "Confusion matrix (test)", 1),
                                spin(plotlyOutput(if (reg) "plt_avp" else "plt_cm", height = "400px"))),
                           card(full_screen = TRUE, ch("🏋️", "Train vs test", 2), plotlyOutput("plt_tt", height = "230px"), tableOutput("tbl_tt"))),
            layout_columns(col_widths = c(6, 6),
                           card(full_screen = TRUE, ch(if (reg) "📉" else if (length(fit$rec$lev) == 2) "📈" else "🎻",
                                                       if (reg) "Residuals vs predicted (±2 SD band)" else if (length(fit$rec$lev) == 2) "ROC curve" else "Probability of the true class", 3),
                                spin(plotlyOutput("plt_diag1", height = "340px"))),
                           card(full_screen = TRUE, ch("📊", if (reg) "Residual distribution" else "Per-class precision · recall · F1", 4),
                                spin(plotlyOutput("plt_diag2", height = "340px")))),
            card(full_screen = TRUE, ch("🧾", "Test-set predictions", 5), spin(DTOutput("tbl_preds"))),
            layout_columns(col_widths = c(5, 7),
                           card(full_screen = TRUE, ch("🔁", "Error per repetition", 6), spin(plotlyOutput("plt_err", height = "300px"))),
                           card(ch("🧠", "Model summary", 1), tags$pre(class = "summary", textOutput("mdl_summ", container = span)),
                                div(class = "d-flex flex-wrap gap-2 mt-2",
                                    downloadButton("dl_preds", "Test predictions", class = "btn-soft btn-outline-primary btn-sm"),
                                    downloadButton("dl_model", "Model (.rds)", class = "btn-soft btn-outline-primary btn-sm"),
                                    downloadButton("dl_summ", "Summary (.txt)", class = "btn-soft btn-outline-primary btn-sm")))))
  })
  observeEvent(input$go_ai, nav_select("nav", "ai"))
  
  output$plt_avp <- renderPlotly({
    fit <- rv$fit; req(fit, fit$rec$type == "regression"); hue <- HUE()
    a <- fit$test$actual; p <- fit$test$pred; lim <- range(c(a, p)); err <- abs(p - a)
    fitl <- stats::lm(p ~ a); xs <- seq(lim[1], lim[2], length.out = 50)
    plot_ly(type = "scatter", mode = "lines") |>
      add_trace(x = lim, y = lim, mode = "lines", name = "Perfect prediction", line = list(color = "#94A3B8", dash = "dash", width = 2)) |>
      add_trace(x = xs, y = predict(fitl, data.frame(a = xs)), mode = "lines", name = "Fitted trend", line = list(color = SERIES[2], width = 3)) |>
      add_trace(x = a, y = p, mode = "markers", name = "Test observations",
                marker = list(size = 11, color = err, colorscale = seq_scale(hue), showscale = TRUE, colorbar = list(title = "|error|", thickness = 12),
                              line = list(color = "white", width = 1.5), opacity = .9),
                text = sprintf("Observed %.4g<br>Predicted %.4g<br>Error %.4g", a, p, p - a), hoverinfo = "text") |>
      pl(paste("Observed", fit$rec$target), paste("Predicted", fit$rec$target))
  })
  
  output$plt_cm <- renderPlotly({
    fit <- rv$fit; req(fit, fit$rec$type == "classification"); hue <- HUE()
    cm <- fit$test$metrics$cm; lev <- rownames(cm)
    pct <- sweep(cm, 1, pmax(rowSums(cm), 1), "/")
    txt <- matrix(sprintf("<b>%d</b><br>%.0f%%", cm, 100 * pct), nrow(cm))
    plot_ly(x = lev, y = lev, z = unclass(pct), type = "heatmap", text = txt, texttemplate = "%{text}",
            colorscale = seq_scale(hue), zmin = 0, zmax = 1, xgap = 5, ygap = 5, textfont = list(size = 15),
            colorbar = list(title = "Row %", thickness = 12, tickformat = ".0%"),
            hovertemplate = "Actual %{y}<br>Predicted %{x}<br>%{text}<extra></extra>") |>
      pl("Predicted", "Actual", legend = FALSE) |> layout(yaxis = list(autorange = "reversed"))
  })
  
  tt_data <- reactive({
    fit <- rv$fit; req(fit)
    if (fit$rec$type == "regression") { a <- fit$trainset$metrics; b <- fit$test$metrics
    data.frame(Metric = c("RMSE", "MAE", "R²", "r", "MAPE %"), Train = as.numeric(a), Test = as.numeric(b)) }
    else { a <- fit$trainset$metrics$overall; b <- fit$test$metrics$overall
    d <- data.frame(Metric = c("Accuracy", "Cohen's κ", "Balanced acc.", "Macro F1", "ROC AUC"), Train = as.numeric(a), Test = as.numeric(b)); d[!is.na(d$Test), ] }
  })
  output$tbl_tt <- renderTable({ d <- tt_data(); d$Train <- signif(d$Train, 4); d$Test <- signif(d$Test, 4); d }, striped = TRUE, hover = TRUE, width = "100%", na = "—")
  output$plt_tt <- renderPlotly({
    d <- tt_data(); fit <- rv$fit
    d <- if (fit$rec$type == "regression") d[d$Metric %in% c("R²", "r"), ] else d
    plot_ly(d, x = ~Metric, y = ~Train, type = "bar", name = "Train", marker = list(color = SERIES[1], line = list(color = "white", width = 2)),
            text = ~signif(Train, 3), textposition = "outside") |>
      add_trace(y = ~Test, name = "Test", marker = list(color = SERIES[2], line = list(color = "white", width = 2)), text = ~signif(Test, 3)) |>
      pl(NULL, NULL) |> layout(barmode = "group", bargap = .35, xaxis = list(title = ""), yaxis = list(title = "", range = c(min(0, min(d$Test, d$Train, na.rm = TRUE)), 1.15)),
                               legend = list(orientation = "h", y = 1.15, x = 0), margin = list(t = 10))
  })
  
  output$plt_diag1 <- renderPlotly({
    fit <- rv$fit; req(fit); hue <- HUE()
    if (fit$rec$type == "regression") {
      r <- fit$test$actual - fit$test$pred; p <- fit$test$pred; s <- stats::sd(r); xr <- range(p)
      plot_ly(type = "scatter", mode = "lines") |>
        add_trace(x = c(xr, rev(xr)), y = c(2 * s, 2 * s, -2 * s, -2 * s), fill = "toself", fillcolor = alpha_col(hue, .1),
                  line = list(width = 0), name = "±2 SD", hoverinfo = "skip") |>
        add_trace(x = xr, y = c(0, 0), mode = "lines", name = "Zero", line = list(color = DIV_NEG, dash = "dash")) |>
        add_trace(x = p, y = r, mode = "markers", name = "Residual",
                  marker = list(size = 10, color = ifelse(abs(r) > 2 * s, SERIES[2], hue), line = list(color = "white", width = 1.5), opacity = .85),
                  hovertemplate = "Predicted %{x:.4g}<br>Residual %{y:.4g}<extra></extra>") |>
        pl("Predicted", "Residual (observed − predicted)")
    } else if (length(fit$rec$lev) == 2) {
      y <- as.numeric(fit$test$actual == fit$rec$lev[2]); s <- fit$test$prob[, 2]
      th <- sort(unique(c(-Inf, s, Inf)), decreasing = TRUE)
      roc <- t(sapply(th, function(t) c(fpr = mean(s[y == 0] >= t), tpr = mean(s[y == 1] >= t))))
      plot_ly(type = "scatter", mode = "lines") |>
        add_trace(x = c(0, 1), y = c(0, 1), line = list(color = "#94A3B8", dash = "dash"), name = "Chance") |>
        add_trace(x = roc[, "fpr"], y = roc[, "tpr"], fill = "tozeroy", fillcolor = alpha_col(hue, .2),
                  line = list(color = hue, width = 4, shape = "hv"), name = sprintf("AUC = %.3f", fit$test$metrics$overall["AUC"])) |>
        pl("False positive rate", "True positive rate") |> layout(legend = list(x = .55, y = .1, orientation = "v"))
    } else {
      lev <- fit$rec$lev; cols <- level_cols(lev)
      pt <- fit$test$prob[cbind(seq_along(fit$test$actual), as.integer(fit$test$actual))]
      p <- plot_ly()
      for (l in lev) { k <- fit$test$actual == l
      p <- add_trace(p, x = rep(l, sum(k)), y = pt[k], type = "violin", name = l, box = list(visible = TRUE), points = "all", jitter = .4, pointpos = 0,
                     fillcolor = alpha_col(cols[[l]], .35), line = list(color = cols[[l]], width = 2), marker = list(color = cols[[l]], size = 5)) }
      pl(p, "Actual class", "P(true class)", legend = FALSE)
    }
  })
  
  output$plt_diag2 <- renderPlotly({
    fit <- rv$fit; req(fit); hue <- HUE()
    if (fit$rec$type == "regression") {
      r <- fit$test$actual - fit$test$pred; xs <- seq(min(r), max(r), length.out = 100)
      bw <- diff(range(r)) / 20
      plot_ly(type = "histogram") |>
        add_trace(x = r, type = "histogram", name = "Residuals", xbins = list(size = bw), marker = list(color = hue, line = list(color = "white", width = 1.5)), opacity = .9) |>
        add_trace(x = xs, y = stats::dnorm(xs, mean(r), stats::sd(r)) * length(r) * bw, type = "scatter", mode = "lines",
                  name = "Normal curve", line = list(color = SERIES[2], width = 3, shape = "spline")) |>
        pl("Residual", "Count")
    } else {
      pcm <- fit$test$metrics$per_class
      plot_ly(pcm, x = ~Class) |>
        add_trace(y = ~Precision, type = "bar", name = "Precision", marker = list(color = SERIES[1], line = list(color = "white", width = 2))) |>
        add_trace(y = ~Recall, type = "bar", name = "Recall", marker = list(color = SERIES[2], line = list(color = "white", width = 2))) |>
        add_trace(y = ~F1, type = "bar", name = "F1", marker = list(color = SERIES[3], line = list(color = "white", width = 2))) |>
        pl(NULL, NULL) |> layout(barmode = "group", bargap = .3, yaxis = list(range = c(0, 1.05), tickformat = ".0%"))
    }
  })
  
  output$tbl_preds <- renderDT({
    fit <- rv$fit; req(fit)
    d <- fit$raw_test[, setdiff(names(fit$raw_test), fit$rec$target), drop = FALSE]
    if (fit$rec$type == "regression") {
      out <- data.frame(Actual = signif(fit$test$actual, 5), Predicted = signif(fit$test$pred, 5), Error = signif(fit$test$pred - fit$test$actual, 4), d, check.names = FALSE)
      mx <- max(abs(out$Error))
      datatable(out, rownames = FALSE, class = "compact hover stripe", options = list(scrollX = TRUE, pageLength = 8, dom = "ftip")) |>
        formatStyle("Error", background = styleColorBar(c(-mx, mx), alpha_col(SERIES[2], .45)), backgroundSize = "95% 70%", backgroundRepeat = "no-repeat", backgroundPosition = "center")
    } else {
      conf <- apply(fit$test$prob, 1, max)
      out <- data.frame(Actual = as.character(fit$test$actual), Predicted = as.character(fit$test$pred),
                        Result = ifelse(fit$test$actual == fit$test$pred, "✔ correct", "✖ wrong"), Confidence = round(conf, 3), d, check.names = FALSE)
      datatable(out, rownames = FALSE, class = "compact hover stripe", options = list(scrollX = TRUE, pageLength = 8, dom = "ftip")) |>
        formatStyle("Result", color = styleEqual(c("✔ correct", "✖ wrong"), c("#0ca30c", "#d03b3b")), fontWeight = "bold") |>
        formatStyle("Confidence", background = styleColorBar(c(0, 1), alpha_col(SERIES[1], .4)), backgroundSize = "95% 70%", backgroundRepeat = "no-repeat", backgroundPosition = "center")
    }
  })
  
  output$plt_err <- renderPlotly({
    fit <- rv$fit; req(fit)
    e <- fit$train$model$result.matrix["error", ]
    plot_ly(x = paste("Rep", seq_along(e)), y = e, type = "bar",
            marker = list(color = ifelse(seq_along(e) == fit$train$best, SERIES[3], "#CBD5E1"), line = list(color = "white", width = 2)),
            text = signif(e, 4), textposition = "outside", hovertemplate = "%{x}: %{y:.5g}<extra></extra>") |>
      pl(NULL, "Training error (best = green)", legend = FALSE) |>
      layout(yaxis = list(range = c(0, max(e) * 1.25)), bargap = if (length(e) == 1) .7 else .35)
  })
  
  summary_text <- function(fit) {
    tr <- fit$train
    c("NeuroLab v8 — model summary", strrep("─", 40),
      paste("Dataset      :", fit$dataset),
      paste("Task         :", fit$rec$type, if (!is.null(fit$rec$lev)) paste0("(", paste(fit$rec$lev, collapse = ", "), ")") else ""),
      paste("Target       :", fit$rec$target), paste("Inputs       :", length(fit$rec$features)),
      paste("Hidden       :", paste(tr$hidden, collapse = " → ")), paste("Activation   :", ACT_LABELS[[tr$act]]),
      paste("Algorithm    :", tr$algorithm, if (tr$algorithm == "backprop") paste("(lr", tr$lr, ")") else ""),
      paste("Error fn     :", tr$err),
      paste("Threshold    :", tr$threshold, if (tr$threshold > tr$threshold_req) paste("(requested", tr$threshold_req, ")") else ""),
      paste("Reps         :", tr$reps_ok, "converged of", tr$reps_req, "· best =", tr$best),
      paste("Steps (best) :", format(tr$steps, big.mark = ",")), paste("Train error  :", signif(tr$train_error, 6)),
      paste("Weights      :", length(unlist(tr$model$weights[[tr$best]]))),
      "", "Features:", paste(" ", fit$rec$features),
      if (length(fit$rec$dropped)) c("", "Dropped columns:", paste(" ", fit$rec$dropped)))
  }
  output$mdl_summ <- renderText({ req(rv$fit); paste(summary_text(rv$fit), collapse = "\n") })
  
  output$dl_preds <- downloadHandler(
    filename = function() paste0("neurolab_test_predictions_", Sys.Date(), ".csv"),
    content = function(f) {
      fit <- rv$fit; out <- fit$raw_test
      out$Actual <- as.character(fit$test$actual); out$Predicted <- as.character(fit$test$pred)
      if (fit$rec$type == "regression") out$Residual <- fit$test$actual - fit$test$pred
      else { pr <- round(fit$test$prob, 5); colnames(pr) <- paste0("P_", colnames(pr)); out <- cbind(out, pr) }
      utils::write.csv(out, f, row.names = FALSE)
    })
  output$dl_model <- downloadHandler(filename = function() paste0("neurolab_model_", Sys.Date(), ".rds"),
                                     content = function(f) saveRDS(list(rec = rv$fit$rec, train = rv$fit$train), f))
  output$dl_summ <- downloadHandler(filename = function() paste0("neurolab_summary_", Sys.Date(), ".txt"),
                                    content = function(f) writeLines(summary_text(rv$fit), f))
  
  # ── INTERPRETATION ────────────────────────────────────────────────────────
  output$interp_ui <- renderUI({
    fit <- rv$fit; if (is.null(fit)) return(no_model())
    L <- length(fit$train$model$weights[[fit$train$best]])
    vars <- if (!is.null(fit$perm)) fit$perm$Feature[order(-fit$perm$Importance)] else fit$rec$vars
    tagList(
      card(full_screen = TRUE,
           ch("🌐", "Network", 1, div(class = "ms-auto d-flex gap-3 align-items-center small fw-normal",
                                      radioButtons("net_view", NULL, c("Diagram" = "net", "Sankey flow" = "sankey"), inline = TRUE),
                                      span(style = paste0("color:", DIV_POS, ";font-weight:700"), "━ positive"), span(style = paste0("color:", DIV_NEG, ";font-weight:700"), "━ negative"),
                                      checkboxInput("show_bias", "Bias nodes", TRUE, width = "auto"))),
           spin(plotlyOutput("plt_net", height = "500px"))),
      layout_columns(col_widths = c(7, 5),
                     card(full_screen = TRUE, ch("📈", "Partial dependence (PDP + ICE)", 2,
                                                 div(class = "ms-auto d-flex gap-2 align-items-center fw-normal",
                                                     selectInput("pdp_var", NULL, vars, vars[1], width = "190px"), checkboxInput("pdp_ice", "ICE lines", TRUE, width = "auto"))),
                          spin(plotlyOutput("plt_pdp", height = "380px")),
                          div(class = "small text-secondary", "How the prediction changes as one variable moves across its range, averaging over the other variables. Thin lines = individual observations (ICE).")),
                     card(full_screen = TRUE, ch("🧪", "Permutation importance", 3), spin(plotlyOutput("plt_perm", height = "380px")),
                          div(class = "small text-secondary", "Loss in test performance when a variable is shuffled — the most reliable importance measure here."))),
      layout_columns(col_widths = c(5, 7),
                     card(full_screen = TRUE, ch("🔗", "Olden connection-weight importance", 4), spin(plotlyOutput("plt_olden", height = "380px"))),
                     card(full_screen = TRUE, ch("🔥", "Weight matrix", 5,
                                                 div(class = "ms-auto", selectInput("wt_layer", NULL, setNames(seq_len(L), c("Input → H1", if (L > 2) paste0("H", 1:(L - 2), " → H", 2:(L - 1)),
                                                                                                                             paste0(if (L > 1) paste0("H", L - 1) else "Input", " → Output"))[seq_len(L)]), width = "180px"))),
                          spin(plotlyOutput("plt_wt", height = "380px")))),
      card(full_screen = TRUE, ch("🎻", "Weight distribution by layer", 6), spin(plotlyOutput("plt_wdist", height = "300px"))))
  })
  
  output$plt_net <- renderPlotly({
    req(rv$fit)
    if (identical(input$net_view, "sankey")) sankey_plot(rv$fit) else network_plot(rv$fit, isTRUE(input$show_bias %||% TRUE))
  })
  output$plt_pdp <- renderPlotly({
    fit <- rv$fit; req(fit, input$pdp_var, input$pdp_var %in% fit$rec$vars)
    pd <- tryCatch(pdp_data(fit, input$pdp_var), error = function(e) NULL); req(pd)
    pdp_plot(pd, HUE(), isTRUE(input$pdp_ice))
  })
  output$plt_perm <- renderPlotly({ req(rv$fit); importance_plot(rv$fit$perm, if (rv$fit$rec$type == "regression") "Increase in RMSE" else "Drop in accuracy", HUE()) })
  output$plt_olden <- renderPlotly({ req(rv$fit); importance_plot(rv$fit$olden, "Olden importance", SERIES[7]) })
  output$plt_wt <- renderPlotly({
    fit <- rv$fit; req(fit, input$wt_layer)
    W <- fit$train$model$weights[[fit$train$best]]; i <- as.integer(input$wt_layer); req(i <= length(W))
    w <- W[[i]]; L <- length(W)
    rowlab <- c("Bias", if (i == 1) fit$rec$features else paste0("H", i - 1, ".", seq_len(nrow(w) - 1)))
    collab <- if (i == L) { if (fit$rec$type == "regression") fit$rec$target else if (length(fit$rec$lev) == 2) paste0("P(", fit$rec$lev[2], ")") else fit$rec$lev }
    else paste0("H", i, ".", seq_len(ncol(w)))
    mx <- max(abs(w))
    plot_ly(x = collab, y = rowlab, z = w, type = "heatmap", zmin = -mx, zmax = mx, colorscale = div_scale, xgap = 2, ygap = 2,
            text = round(w, 2), texttemplate = if (length(w) <= 300) "%{text}" else "", colorbar = list(title = "w", thickness = 12),
            hovertemplate = "%{y} → %{x}<br>w = %{z:.4f}<extra></extra>") |>
      pl(NULL, NULL, legend = FALSE) |> layout(yaxis = list(autorange = "reversed"))
  })
  output$plt_wdist <- renderPlotly({
    fit <- rv$fit; req(fit); W <- fit$train$model$weights[[fit$train$best]]; L <- length(W)
    labs <- c(if (L > 1) paste0(c("Input", paste0("H", seq_len(L - 2))), " → H", seq_len(L - 1)), paste0(if (L > 1) paste0("H", L - 1) else "Input", " → Out"))
    cols <- level_cols(labs); p <- plot_ly()
    for (i in seq_len(L)) p <- add_trace(p, x = as.vector(W[[i]]), y = rep(labs[i], length(W[[i]])), type = "violin", orientation = "h", name = labs[i],
                                         side = "positive", points = "all", jitter = .3, pointpos = -.3, fillcolor = alpha_col(cols[[i]], .35),
                                         line = list(color = cols[[i]], width = 2), marker = list(color = cols[[i]], size = 4, opacity = .6), meanline = list(visible = TRUE))
    pl(p, "Weight value", NULL, legend = FALSE)
  })
  
  # ── AI INSIGHTS & REPORT ──────────────────────────────────────────────────
  output$ai_ui <- renderUI({
    fit <- rv$fit; ins <- rv$ins; if (is.null(fit)) return(no_model())
    if (is.null(ins)) return(div(class = "note note-e", "The insight engine could not interpret this model."))
    st_lab <- c(good = "✔ Good", warn = "⚠ Attention", bad = "✖ Concern", info = "ℹ Note")
    secs <- ins$sections[names(ins$sections) != "summary"]
    cards <- lapply(seq_along(secs), function(i) { s <- secs[[i]]
    card(class = paste0("ins ins-", s$status), ch(s$icon, s$title, (i - 1) %% 6 + 1, span(class = "st-pill", st_lab[[s$status]])),
         card_body(lapply(s$paras, p), if (length(s$bullets)) tags$ul(lapply(s$bullets, tags$li)))) })
    layout_columns(col_widths = c(8, 4),
                   div(div(class = "exec", h4("✨ Executive summary"), p(ins$sections$summary$paras)), cards),
                   div(
                     card(ch("📄", "Word report", 6),
                          if (!HAS_OFFICER) div(class = "note note-e", "Install the report packages first:", tags$code("install.packages(c('officer','flextable'))")),
                          textInput("rep_title", "Report title", "Neural Network Analysis Report"),
                          textInput("rep_author", "Author", ""),
                          checkboxGroupInput("rep_sections", "Include",
                                             c("Model at a glance (tables)" = "metrics", "AI interpretation" = "ai", "Diagnostic figures" = "figures",
                                               "Interpretation figures (importance, PDP, network)" = "interp", "Tuning results" = "tune", "Appendix" = "appendix"),
                                             selected = c("metrics", "ai", "figures", "interp", "tune", "appendix")),
                          div(class = "small text-secondary mb-2", "Figures use the current colour theme (", TH()$name, ")."),
                          downloadButton("dl_word", "Download Word report (.docx)", class = "btn-glow btn-word")),
                     card(ch("🤖", "Claude AI narrative (optional)", 4),
                          p(class = "small text-secondary", "Ask Claude (Anthropic) to write an expert narrative from the computed statistics. It is added to the Word report. Only summary statistics and variable names are sent — never your raw data."),
                          passwordInput("ai_key", "Anthropic API key", placeholder = if (nzchar(Sys.getenv("ANTHROPIC_API_KEY"))) "Using ANTHROPIC_API_KEY from environment" else "sk-ant-…"),
                          textInput("ai_model", "Model", "claude-sonnet-5-5"),
                          if (!HAS_HTTR2) div(class = "note note-w", "Needs package httr2:", tags$code("install.packages('httr2')")),
                          actionButton("ai_go", "✨  Generate narrative", class = "btn-glow btn-ai"),
                          uiOutput("ai_out"))))
  })
  
  observeEvent(input$ai_go, {
    on.exit(done_overlay(), add = TRUE)
    req(rv$fit, rv$ins)
    key <- if (nzchar(input$ai_key %||% "")) input$ai_key else Sys.getenv("ANTHROPIC_API_KEY")
    res <- tryCatch(claude_narrative(rv$fit, rv$ins, key, input$ai_model %||% "claude-sonnet-5-5"), error = function(e) e)
    if (inherits(res, "error")) { notify(paste("<b>Claude request failed:</b>", htmltools::htmlEscape(conditionMessage(res))), "error", 10); return() }
    rv$ai <- res; notify("✨ Narrative ready — it will be included in the Word report.")
  })
  output$ai_out <- renderUI({ req(rv$ai); div(class = "ai-out", md_lite_html(rv$ai)) })
  
  output$dl_word <- downloadHandler(
    filename = function() paste0("NeuroLab_report_", gsub("[^A-Za-z0-9]+", "_", rv$fit$dataset %||% "model"), "_", Sys.Date(), ".docx"),
    content = function(f) {
      withProgress(message = "Building Word report…", value = .3, {
        build_word_report(f, rv$fit, rv$ins, rv$ai, rv$tune, TH(), title = input$rep_title %||% "Neural Network Analysis Report",
                          author = input$rep_author %||% "", sections = input$rep_sections %||% c("metrics", "ai", "figures", "interp"))
      })
    })
  
  # ── TUNING ────────────────────────────────────────────────────────────────
  tune_grid <- reactive({
    nn <- suppressWarnings(as.integer(trimws(strsplit(input$t_neurons, ",")[[1]])))
    nn <- unique(nn[is.finite(nn) & nn > 0 & nn <= 64])
    ly <- as.integer(input$t_layers %||% 1); ac <- input$t_act %||% "logistic"
    lr <- if (input$algo == "backprop") { v <- suppressWarnings(as.numeric(trimws(strsplit(input$t_lr %||% "0.01", ",")[[1]]))); unique(v[is.finite(v) & v > 0]) } else NA
    if (!length(nn) || !length(lr)) return(NULL)
    expand.grid(Neurons = nn, Layers = ly, Activation = ac, LR = lr, stringsAsFactors = FALSE)
  })
  
  output$tune_size <- renderUI({
    g <- tune_grid(); if (is.null(g)) return(div(class = "note note-e", "Enter valid neuron counts (1–64)."))
    fits <- nrow(g) * as.integer(input$t_folds)
    div(class = paste("note", if (fits > 60) "note-w" else "note-i"),
        sprintf("%d configurations × %s = %d model fits. Algorithm: %s (from Build tab).", nrow(g),
                if (input$t_folds == "1") "1 holdout" else paste(input$t_folds, "folds"), fits, input$algo))
  })
  
  observeEvent(input$btn_tune, {
    on.exit(done_overlay(), add = TRUE)
    ds <- tryCatch(dataset(), error = function(e) NULL); req(ds, input$target); g <- tune_grid()
    if (is.null(g)) { notify("Invalid grid values.", "warning"); return() }
    folds <- as.integer(input$t_folds); res <- vector("list", nrow(g))
    withProgress(message = "Grid search", value = 0, {
      for (i in seq_len(nrow(g))) {
        incProgress(1 / nrow(g), detail = sprintf("%d/%d · %d×%d %s", i, nrow(g), g$Layers[i], g$Neurons[i], g$Activation[i]))
        r <- tryCatch(cv_score(ds$df, input$target, input$ptype, input$exclude %||% character(), input$scaling,
                               rep(g$Neurons[i], g$Layers[i]), g$Activation[i], input$algo,
                               if (is.na(g$LR[i])) 0.01 else g$LR[i], input$t_thresh, input$t_steps, folds, input$seed),
                      error = function(e) structure(c(score = NA, steps = NA, secs = NA), msg = conditionMessage(e)))
        res[[i]] <- cbind(g[i, ], Score = r[["score"]], Steps = round(r[["steps"]]), Seconds = round(r[["secs"]], 2),
                          Issue = attr(r, "msg") %||% "", stringsAsFactors = FALSE)
      }
    })
    out <- do.call(rbind, res)
    tp <- if (input$ptype == "auto") detect_type(ds$df[[input$target]]) else input$ptype
    attr(out, "metric") <- if (tp == "regression") "RMSE" else "Accuracy"
    attr(out, "higher") <- tp != "regression"
    if (all(is.na(out$Score))) { notify("No configuration converged. Raise the threshold or Max steps.", "error"); return() }
    rv$tune <- out
    notify(sprintf("✅ Grid search done — %d of %d configurations succeeded.", sum(!is.na(out$Score)), nrow(out)))
  })
  
  tune_best <- reactive({
    r <- rv$tune; req(r); ok <- r[!is.na(r$Score), ]
    sc <- if (attr(r, "higher")) -ok$Score else ok$Score
    ok[order(signif(sc, 6), ok$Layers * ok$Neurons, ok$Steps)[1], ]
  })
  
  output$tune_best <- renderUI({
    r <- rv$tune; if (is.null(r)) return(NULL); b <- tune_best()
    div(class = "note note-ok mt-3", tags$b("🏆 Best configuration"), br(),
        span(class = "chip chip-v", paste(b$Layers, "layer(s) ×", b$Neurons, "neurons")), span(class = "chip chip-t", ACT_LABELS[[b$Activation]]),
        if (!is.na(b$LR)) span(class = "chip chip-a", paste("LR", b$LR)),
        span(class = "chip chip-p", sprintf("%s = %.4g", attr(r, "metric"), b$Score)))
  })
  output$tune_apply_ui <- renderUI({ req(rv$tune); actionButton("btn_apply", "↩️ Use best config in Build tab", class = "btn btn-outline-primary btn-soft w-100 mt-2") })
  observeEvent(input$btn_apply, {
    b <- tune_best()
    rv$pending_hidden <- rep(b$Neurons, b$Layers)
    updateNumericInput(session, "n_layers", value = b$Layers)
    updateSelectInput(session, "act", selected = b$Activation)
    if (!is.na(b$LR)) updateNumericInput(session, "lr", value = b$LR)
    updateSelectInput(session, "thresh", selected = input$t_thresh)
    nav_select("nav", "build")
    notify("Best configuration applied — press <b>Train network</b>.")
  })
  
  output$plt_heat <- renderPlotly({
    r <- rv$tune; if (is.null(r)) return(empty_plot("Run a grid search to see results")); hue <- HUE()
    r$Config <- paste0(r$Layers, "L · ", r$Activation, ifelse(is.na(r$LR), "", paste0(" · lr ", r$LR)))
    cfgs <- unique(r$Config); nn <- sort(unique(r$Neurons))
    z <- matrix(sapply(cfgs, function(cf) sapply(nn, function(n) { v <- r$Score[r$Config == cf & r$Neurons == n]; if (length(v)) v[1] else NA })), nrow = length(nn))
    hi <- attr(r, "higher"); cs <- seq_scale(hue); if (!hi) cs <- lapply(seq_along(cs), function(i) list(cs[[i]][[1]], cs[[length(cs) - i + 1]][[2]]))
    plot_ly(x = cfgs, y = paste(nn, "neurons"), z = z, type = "heatmap", colorscale = cs, xgap = 4, ygap = 4,
            text = ifelse(is.na(z), "—", formatC(z, digits = 4, format = "g")), texttemplate = "<b>%{text}</b>", textfont = list(size = 13),
            hovertemplate = "%{x}<br>%{y}<br>score %{z:.4g}<extra></extra>", colorbar = list(title = attr(r, "metric"), thickness = 12)) |>
      pl(NULL, NULL, legend = FALSE, title = list(text = paste(attr(r, "metric"), if (hi) "(darker = better)" else "(darker = better, lower RMSE)"), font = list(size = 13)))
  })
  
  output$plt_land <- renderPlotly({
    r <- rv$tune; if (is.null(r)) return(empty_plot("Run a grid search to see results"))
    r <- r[!is.na(r$Score), ]; acts <- unique(r$Activation); cols <- level_cols(acts)
    syms <- c("circle", "diamond", "square", "triangle-up", "star")
    p <- plot_ly()
    for (a in acts) for (L in sort(unique(r$Layers))) { d <- r[r$Activation == a & r$Layers == L, ]; if (!nrow(d)) next; d <- d[order(d$Neurons), ]
    p <- add_trace(p, x = d$Neurons, y = d$Score, type = "scatter", mode = "lines+markers", name = paste0(a, " · ", L, "L"),
                   line = list(color = cols[[a]], width = 2.5, dash = c("solid", "dash", "dot", "dashdot", "longdash")[min(L, 5)]),
                   marker = list(size = 13 + pmin(10, d$Seconds), color = cols[[a]], symbol = syms[min(L, 5)], line = list(color = "white", width = 2)),
                   text = sprintf("%s · %d layer(s) · %d neurons<br>%s = %.4g<br>%.1f s", a, L, d$Neurons, attr(rv$tune, "metric"), d$Score, d$Seconds), hoverinfo = "text") }
    pl(p, "Neurons per layer", attr(rv$tune, "metric"))
  })
  
  output$plt_rank <- renderPlotly({
    r <- rv$tune; if (is.null(r)) return(empty_plot("Run a grid search to see results"))
    r <- r[!is.na(r$Score), ]
    r$Label <- paste0(r$Layers, "×", r$Neurons, " ", r$Activation, ifelse(is.na(r$LR), "", paste0(" lr", r$LR)))
    r <- r[order(r$Score, decreasing = !attr(rv$tune, "higher")), ]; r$Label <- factor(r$Label, levels = r$Label)
    plot_ly(r, x = ~Score, y = ~Label, type = "bar", orientation = "h",
            marker = list(color = c(rep("#CBD5E1", nrow(r) - 1), HUE()), line = list(color = "white", width = 2)),
            text = ~formatC(Score, digits = 4, format = "g"), textposition = "outside", hovertemplate = "%{y}: %{x:.4g}<extra></extra>") |>
      pl(attr(rv$tune, "metric"), NULL, legend = FALSE)
  })
  
  output$tbl_tune <- renderDT({
    r <- rv$tune; req(r); r <- r[order(r$Score, decreasing = attr(r, "higher"), na.last = TRUE), ]
    names(r)[names(r) == "Score"] <- attr(rv$tune, "metric")
    datatable(r, rownames = FALSE, class = "compact hover stripe", options = list(dom = "tip", pageLength = 12)) |>
      formatRound(attr(rv$tune, "metric"), 4) |>
      formatStyle(attr(rv$tune, "metric"), background = styleColorBar(range(r[[attr(rv$tune, "metric")]], na.rm = TRUE), alpha_col(HUE(), .4)),
                  backgroundSize = "95% 70%", backgroundRepeat = "no-repeat", backgroundPosition = "center")
  })
  output$dl_tune <- downloadHandler(filename = function() paste0("neurolab_tuning_", Sys.Date(), ".csv"),
                                    content = function(f) utils::write.csv(rv$tune, f, row.names = FALSE))
  
  # ── PREDICT ───────────────────────────────────────────────────────────────
  output$predict_ui <- renderUI({
    fit <- rv$fit; if (is.null(fit)) return(no_model()); rec <- fit$rec
    inputs <- c(
      lapply(names(rec$num), function(v) { r <- rec$num[[v]]
      sliderInput(paste0("wi_", make.names(v)), v, min = signif(r$min, 4), max = signif(r$max, 4), value = signif(r$median, 4),
                  step = signif((r$max - r$min) / 100, 2)) }),
      lapply(names(rec$cat), function(v) selectInput(paste0("wi_", make.names(v)), v, rec$cat[[v]]$levels, rec$cat[[v]]$mode)))
    layout_columns(col_widths = c(7, 5),
                   card(ch("🎛️", "What-if predictor", 1, span(class = "ms-auto small text-secondary fw-normal", "drag the sliders — prediction updates live")),
                        div(style = "display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:0 18px", inputs)),
                   div(card(ch("🔮", "Live prediction", 4), uiOutput("whatif_out")),
                       card(ch("📂", "Batch scoring", 2),
                            div(class = "note note-i", "Upload a file with these columns (target not needed): ", tags$b(paste(rec$vars, collapse = ", "))),
                            downloadButton("dl_template", "Download template", class = "btn-soft btn-outline-secondary btn-sm mb-2"),
                            fileInput("score_file", NULL, accept = c(".csv", ".tsv", ".txt"), buttonLabel = "Browse…"),
                            actionButton("btn_score", "🔮 Score file", class = "btn-glow btn-cool"),
                            uiOutput("score_status"), DTOutput("tbl_score"), uiOutput("dl_score_ui"))))
  })
  
  output$whatif_out <- renderUI({
    fit <- rv$fit; req(fit); rec <- fit$rec
    vals <- lapply(rec$vars, function(v) input[[paste0("wi_", make.names(v))]])
    if (any(vapply(vals, is.null, TRUE))) return(NULL)
    nd <- as.data.frame(setNames(vals, rec$vars), stringsAsFactors = FALSE, check.names = FALSE)
    out <- tryCatch(nl_predict_df(fit, nd), error = function(e) NULL); req(out)
    if (rec$type == "regression") {
      lo <- rec$y$min; hi <- rec$y$min + rec$y$range; pos <- 100 * (out$pred - lo) / (hi - lo)
      tagList(div(class = "pred-result", div(class = "small fw-bold text-uppercase", paste("Predicted", rec$target)),
                  div(class = "v", signif(out$pred, 5)), div(class = "small", sprintf("training range %s – %s", signif(lo, 4), signif(hi, 4)))),
              div(class = "prob-bar mt-3", div(style = sprintf("width:%.1f%%;background:%s", max(2, min(100, pos)), HUE()))))
    } else {
      pr <- out$prob[1, ]; ord <- order(-pr); cols <- level_cols(rec$lev)
      tagList(
        div(class = "pred-result", div(class = "small fw-bold text-uppercase", paste("Predicted", rec$target)),
            div(class = "v", as.character(out$pred)), div(class = "small", sprintf("confidence %.1f%%", 100 * max(pr)))),
        div(class = "mt-3", lapply(ord, function(j) div(class = "prob-row",
                                                        div(class = "d-flex justify-content-between small fw-bold", span(names(pr)[j]), span(sprintf("%.1f%%", 100 * pr[j]))),
                                                        div(class = "prob-bar", div(style = sprintf("width:%.1f%%;background:%s", 100 * pr[j], cols[[names(pr)[j]]])))))))
    }
  })
  
  output$dl_template <- downloadHandler(
    filename = function() "neurolab_scoring_template.csv",
    content = function(f) {
      rec <- rv$fit$rec
      tmp <- as.data.frame(c(lapply(rec$num, function(r) signif(r$median, 5)), lapply(rec$cat, function(r) r$mode)), check.names = FALSE)[, rec$vars, drop = FALSE]
      utils::write.csv(tmp, f, row.names = FALSE)
    })
  
  score_msg <- reactiveVal(NULL)
  observeEvent(input$btn_score, {
    req(rv$fit)
    if (is.null(input$score_file)) { score_msg(list("w", "Choose a file first.")); return() }
    nd <- tryCatch(read_any(input$score_file$datapath), error = function(e) { score_msg(list("e", e$message)); NULL }); req(nd)
    out <- tryCatch(nl_predict_df(rv$fit, nd), error = function(e) { score_msg(list("e", e$message)); NULL }); req(out)
    if (rv$fit$rec$type == "regression") nd[[paste0("Predicted_", rv$fit$rec$target)]] <- round(out$pred, 5)
    else { nd$Predicted_Class <- as.character(out$pred); pr <- round(out$prob, 4); colnames(pr) <- paste0("P_", colnames(pr)); nd <- cbind(nd, pr) }
    rv$score <- nd; score_msg(list("ok", paste("Scored", nrow(nd), "rows.")))
  })
  output$score_status <- renderUI({ m <- score_msg(); if (is.null(m)) return(NULL); div(class = paste0("note mt-2 note-", m[[1]]), m[[2]]) })
  output$tbl_score <- renderDT({ req(rv$score); datatable(rv$score, rownames = FALSE, class = "compact hover stripe", options = list(scrollX = TRUE, pageLength = 6, dom = "tip")) })
  output$dl_score_ui <- renderUI({ req(rv$score); downloadButton("dl_score", "Download scored data", class = "btn-soft btn-outline-primary btn-sm mt-2") })
  output$dl_score <- downloadHandler(filename = function() paste0("neurolab_scored_", Sys.Date(), ".csv"),
                                     content = function(f) utils::write.csv(rv$score, f, row.names = FALSE))
}

shinyApp(ui, server)