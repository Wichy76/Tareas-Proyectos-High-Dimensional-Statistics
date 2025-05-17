load("proy1hd2.RData")   # X (n×p) y Y (n×1)

library(glmnet)
library(ggplot2)
library(doParallel)

X <- scale(X, center = TRUE, scale = FALSE)
Y <- as.numeric(Y - mean(Y))

fb_giraud <- function(X, Y, sigma, K, d0, max_iter = 100) {
  n <- nrow(X); p <- ncol(X)
  XtX  <- crossprod(X); XtY  <- crossprod(X, Y); YtY  <- sum(Y^2)
  Cc   <- p * log(1 + 1/p); logp <- log(p)
  crit <- function(idx) {
    d_m <- length(idx)
    rss <- if (d_m==0) YtY else {
      S <- XtX[idx, idx, drop=FALSE]; v <- XtY[idx]; β <- solve(S, v)
      YtY - sum(β * v)
    }
    pen <- K * (sqrt(d_m) + sqrt(2*(Cc + d_m*logp)))^2
    rss + sigma^2 * pen
  }
  m_curr    <- order(apply(X,2,var), decreasing=TRUE)[1:d0]
  crit_curr <- crit(m_curr)
  for (it in seq_len(max_iter)) {
    improved <- FALSE
    best_in   <- NULL; best_crit <- crit_curr
    for (j in setdiff(seq_len(p), m_curr)) {
      c_try <- crit(c(m_curr, j))
      if (c_try < best_crit) { best_crit <- c_try; best_in <- j }
    }
    if (!is.null(best_in)) {
      m_curr    <- c(m_curr, best_in); crit_curr <- best_crit; improved <- TRUE
    }
    best_out  <- NULL; best_crit <- crit_curr
    for (j in m_curr) {
      c_try <- crit(setdiff(m_curr, j))
      if (c_try < best_crit) { best_crit <- c_try; best_out <- j }
    }
    if (!is.null(best_out)) {
      m_curr    <- setdiff(m_curr, best_out); crit_curr <- best_crit; improved <- TRUE
    }
    if (!improved) break
  }
  beta <- numeric(p)
  if (length(m_curr)>0) {
    S <- XtX[m_curr, m_curr, drop=FALSE]; v <- XtY[m_curr]
    beta[m_curr] <- solve(S, v)
  }
  list(beta = beta, support = m_curr)
}

fit0       <- lm(Y ~ X + 0)
sigma2_hat <- deviance(fit0) / df.residual(fit0)

p           <- ncol(X)
d0_grid     <- c(floor(p/10), floor(p/5), floor(p/3))
K_const     <- 1.1
Kfold       <- 5
set.seed(2025)
folds       <- sample(rep(1:Kfold, length.out = nrow(X)))
lambda_grid <- 10^seq(-4, 0, length.out = 30)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

mse_fb <- numeric(length(d0_grid))
for (i in seq_along(d0_grid)) {
  d0 <- d0_grid[i]
  mses <- foreach(k = 1:Kfold, .combine = c) %dopar% {
    trn <- which(folds != k); tst <- which(folds == k)
    Xtr <- X[trn, , drop=FALSE]; Ytr <- Y[trn]
    Xts <- X[tst, , drop=FALSE]; Yts <- Y[tst]
    fit_tr   <- lm(Ytr ~ Xtr + 0)
    sigma_tr <- sqrt(deviance(fit_tr) / df.residual(fit_tr))
    res_fb   <- fb_giraud(Xtr, Ytr, sigma_tr, K_const, d0)
    preds    <- Xts %*% res_fb$beta
    mean((Yts - preds)^2)
  }
  mse_fb[i] <- mean(mses)
}

stopCluster(cl)

best_idx_fb   <- which.min(mse_fb)
best_d0       <- d0_grid[best_idx_fb]
best_mse_fb   <- mse_fb[best_idx_fb]

cv_las <- cv.glmnet(X, Y, alpha = 1, lambda = lambda_grid,
                    intercept = TRUE, standardize = TRUE, nfolds = Kfold)
cv_rid <- cv.glmnet(X, Y, alpha = 0, lambda = lambda_grid,
                    intercept = TRUE, standardize = TRUE, nfolds = Kfold)

best_idx_las    <- which.min(cv_las$cvm)
best_lambda_las <- lambda_grid[best_idx_las]
almost_best_lambda_las <- cv_las$lambda.1se
best_mse_las    <- cv_las$cvm[best_idx_las]

best_idx_rid    <- which.min(cv_rid$cvm)
best_lambda_rid <- lambda_grid[best_idx_rid]
almost_best_lambda_rid <- cv_rid$lambda.1se
best_mse_rid    <- cv_rid$cvm[best_idx_rid]

cat("FB mejor d0 =", best_d0, "CV MSE =", round(best_mse_fb, 5), "\n")
cat("Lasso mejor λ =", best_lambda_las, "CV MSE =", round(best_mse_las, 5), "\n")
cat("Lasso mejor λ (1se) =", almost_best_lambda_las, "CV MSE =", round(cv_las$cvm[which(cv_las$lambda == almost_best_lambda_las)], 5), "\n")
cat("Ridge mejor λ =", best_lambda_rid, "CV MSE =", round(best_mse_rid, 5), "\n\n")

res_fb_full   <- fb_giraud(X, Y, sqrt(sigma2_hat), K_const, best_d0)
cat("FB support:", length(res_fb_full$support), "vars ->", sort(res_fb_full$support), "\n")

coef_las_full <- as.matrix(coef(cv_las, s = best_lambda_las))
sel_las       <- which(coef_las_full[-1, ] != 0)
cat("Lasso support:", length(sel_las), "vars ->", sel_las, "\n")

coef_las_full_1se <- as.matrix(coef(cv_las, s = almost_best_lambda_las))
sel_las_1se       <- which(coef_las_full_1se[-1, ] != 0)
cat("Lasso support (1desEst):", length(sel_las_1se), "vars ->", sel_las_1se, "\n")

coef_rid_full <- as.matrix(coef(cv_rid, s = best_lambda_rid))
sel_rid       <- which(coef_rid_full[-1, ] != 0)
cat("Ridge support:", length(sel_rid), "vars ->", sel_rid, "\n\n")

df_fb    <- data.frame(d0 = d0_grid,        MSE = mse_fb)
df_lasso <- data.frame(lambda = lambda_grid, MSE = cv_las$cvm)
df_ridge <- data.frame(lambda = lambda_grid, MSE = cv_rid$cvm)

print(ggplot(df_fb,    aes(x = d0,    y = MSE)) +
        geom_line() + geom_point() + scale_x_log10() +
        labs(title = "FB", x = "d0", y = "MSE") + theme_minimal())

print(ggplot(df_lasso, aes(x = lambda, y = MSE)) +
        geom_line() + scale_x_log10() +
        labs(title = "Lasso", x = expression(lambda), y = "MSE") + theme_minimal())

print(ggplot(df_ridge, aes(x = lambda, y = MSE)) +
        geom_line() + scale_x_log10() +
        labs(title = "Ridge", x = expression(lambda), y = "MSE") + theme_minimal())

