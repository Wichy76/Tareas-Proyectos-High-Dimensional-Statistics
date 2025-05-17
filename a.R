load("proy1hd2.RData")


# ================================================
# Proyecto I 
# ================================================

# 0) Carga de datos y librerías -------------------------------------
# Ajusta la ruta al .RData que contenga X (n×p) y Y (n×1)
# load("ruta/a/tu/proyecto1_workspace.RData")

library(glmnet)
library(ggplot2)

# Punto 1 del enunciado

#Definición de la función Forward–Backward -----------------------

backward_forward <- function(X, Y, sigma, K, d0) {
  n <- nrow(X)
  p <- ncol(X)
  
  crit <- function(idx) {
    # RSS
    if (length(idx) == 0) {
      rss <- sum(Y^2)
    } else {
      fit <- lm(Y ~ X[, idx, drop = FALSE] + 0)
      rss <- sum(resid(fit)^2)
    }
    # penalización según Giraud
    d_m      <- length(idx)
    C_const  <- p * log(1 + 1/p)
    log_invπ <- C_const + d_m * log(p)
    pen      <- K * (sqrt(d_m) + sqrt(2 * log_invπ))^2
    rss + sigma^2 * pen
  }
  
  # inicial: d0 predictores con mayor varianza
  vars   <- apply(X, 2, var)
  m_curr <- order(vars, decreasing = TRUE)[seq_len(d0)]
  
  repeat {
    improved  <- FALSE
    curr_crit <- crit(m_curr)
    
    # paso hacia adelante
    best_in   <- NULL
    best_crit <- curr_crit
    for (j in setdiff(seq_len(p), m_curr)) {
      c_try <- crit(c(m_curr, j))
      if (c_try < best_crit) {
        best_crit <- c_try
        best_in   <- j
      }
    }
    if (!is.null(best_in)) {
      m_curr   <- c(m_curr, best_in)
      curr_crit <- best_crit
      improved <- TRUE
    }
    
    # paso hacia atrás
    best_out  <- NULL
    best_crit <- curr_crit
    for (j in m_curr) {
      c_try <- crit(setdiff(m_curr, j))
      if (c_try < best_crit) {
        best_crit <- c_try
        best_out  <- j
      }
    }
    if (!is.null(best_out)) {
      m_curr   <- setdiff(m_curr, best_out)
      curr_crit <- best_crit
      improved <- TRUE
    }
    
    if (!improved) break
  }
  
  beta <- numeric(p)
  if (length(m_curr) > 0) {
    fit_m       <- lm(Y ~ X[, m_curr, drop = FALSE] + 0)
    beta[m_curr] <- coef(fit_m)
  }
  list(beta = beta, support = m_curr)
}

#  Estimación de σ² -------------------------------------------------
fit0       <- lm(Y ~ X + 0)
sigma2_hat <- deviance(fit0) / df.residual(fit0)

# Parámetros de CV -------------------------------------------------
p         <- ncol(X)
d0_grid   <- c(floor(p/10), floor(p/5), floor(p/3))
K_const   <- 1.1
Kfold     <- 5
set.seed(123)
folds     <- sample(rep(1:Kfold, length.out = nrow(X)))

lambda_grid <- 10^seq(-4, 0, length.out = 30)

# 2) Validación cruzada ----------------------------------------------
# Forward–Backward
mse_fb <- numeric(length(d0_grid))
for (i in seq_along(d0_grid)) {
  d0   <- d0_grid[i]
  mses <- numeric(Kfold)
  for (k in seq_len(Kfold)) {
    trn <- which(folds != k); tst <- which(folds == k)
    Xtr <- X[trn,,drop=FALSE]; Ytr <- Y[trn]
    Xts <- X[tst,,drop=FALSE]; Yts <- Y[tst]
    fit_tr   <- lm(Ytr ~ Xtr + 0)
    sigma_tr <- sqrt(deviance(fit_tr) / df.residual(fit_tr))
    res_fb   <- backward_forward(Xtr, Ytr, sigma = sigma_tr, K = K_const, d0 = d0)
    preds    <- Xts %*% res_fb$beta
    mses[k]  <- mean((Yts - preds)^2)
  }
  mse_fb[i] <- mean(mses)
}
df_fb <- data.frame(d0 = d0_grid, MSE = mse_fb)

# Lasso y Ridge
cv_las   <- cv.glmnet(X, Y, alpha = 1, lambda = lambda_grid,
                      intercept = TRUE, standardize = TRUE, nfolds = Kfold)
df_lasso <- data.frame(lambda = lambda_grid, MSE = cv_las$cvm)

cv_rid   <- cv.glmnet(X, Y, alpha = 0, lambda = lambda_grid,
                      intercept = TRUE, standardize = TRUE, nfolds = Kfold)
df_ridge <- data.frame(lambda = lambda_grid, MSE = cv_rid$cvm)

# Gráficas con eje X en escala log10 -----------------------------
# Forward–Backward
p1 <- ggplot(df_fb, aes(x = d0, y = MSE)) +
  geom_line() + geom_point() +
  scale_x_log10() +
  labs(title = "Forward–Backward", x = "d0 (escala log10)", y = "CV MSE") +
  theme_minimal()

# Lasso
p2 <- ggplot(df_lasso, aes(x = lambda, y = MSE)) +
  geom_line() +
  scale_x_log10() +
  labs(title = "Lasso (α = 1)", x = expression(lambda~"(escala log10)"), y = "CV MSE") +
  theme_minimal()

# Ridge
p3 <- ggplot(df_ridge, aes(x = lambda, y = MSE)) +
  geom_line() +
  scale_x_log10() +
  labs(title = "Ridge (α = 0)", x = expression(lambda~"(escala log10)"), y = "CV MSE") +
  theme_minimal()

# Mostrar gráficas ------------------------------------------------
print(p1)
print(p2)
print(p3)
