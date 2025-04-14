# --------------------------------------------------
# 1. Construcción de la base seno-coseno
# --------------------------------------------------
# Para p = 1 + 2*K, donde:
# - varphi_1(x) = 1
# - varphi_{2k}(x)   = cos(2 pi k x)   para k = 1..K
# - varphi_{2k+1}(x) = sin(2 pi k x)   para k = 1..K
build_design_matrix <- function(x, K = 25) {
  n <- length(x)
  p <- 1 + 2*K        # cantidad total de funciones base
  Phi <- matrix(0, nrow = n, ncol = p)
  
  # varphi_1(x) = 1
  Phi[, 1] <- 1
  
  # Rellenar cosenos y senos
  for (k in 1:K) {
    Phi[, 2*k]   <- cos(2*pi*k*x)
    Phi[, 2*k+1] <- sin(2*pi*k*x)
  }
  Phi
}

# --------------------------------------------------
# 2. Definición de la "verdadera" señal (desconocida)
# --------------------------------------------------
# Puedes ajustar la combinación de senos y cosenos a tu gusto.
ftrue <- function(x) {
  # Un ejemplo de señal "ondulatoria" simple:
  # - 2 senos importantes y 1 cos, para ilustrar
  0.8 + 1.5*sin(2*pi*2*x) - 0.8*cos(2*pi*5*x) + 0.6*sin(2*pi*7*x) 
}

c0 <- 0.5
c_cos <- rep(0.5, 25)
c_sin <- rep(-0.4, 25)

ftrue1 <- function(x) {
  # La función se define como:
  # f(x) = c0 + sum_{k=1}^{25} [ c_cos[k]*cos(2*pi*k*x) + c_sin[k]*sin(2*pi*k*x) ]
  s <- c0
  for(k in 1:25){
    s <- s + c_cos[k]*cos(2*pi*k*x) + c_sin[k]*sin(2*pi*k*x)
  }
  return(s)
}

# --------------------------------------------------
# 3. Generación de datos
# --------------------------------------------------
set.seed(123)  # para reproducibilidad
n <- 60
x <- seq(1, n)/n   # i/n para i=1..n, es decir, x en [1/60, 60/60]
sigma <- 1        # desviación estándar del ruido
# Observaciones ruidosas
Y <- ftrue(x) + rnorm(n, mean = 0, sd = sigma)

# Construimos la matriz X para el modelo lineal
Kmax <- 25
X <- build_design_matrix(x, K = Kmax)
p <- ncol(X)  # p = 1 + 2*Kmax

# --------------------------------------------------
# 4. Implementación del criterio (2.9) y algoritmo backward-forward
# --------------------------------------------------
#   criterio(m) = ||Y - f_m||^2 + sigma^2 * pen(m)
#   pen(m) = K * ( sqrt(dm) + sqrt(2 * log(1/pi_m)) )^2
#   pi_m = (1 + 1/p)^(-p) * p^(-|m|)
#   donde f_m es la proyección de Y sobre S_m
#
# Aquí definimos la función 'backward_forward' y su helper.

backward_forward <- function(X, Y, sigma, K_pen = 1.1, max_iter = 100) {
  n <- length(Y)
  p <- ncol(X)
  
  # Función que evalúa el criterio en un subconjunto m (vector de índices)
  criterio <- function(m) {
    dm <- length(m)
    # pi_m = (1 + 1/p)^(-p) * p^(-dm)
    pi_m <- (1 + 1/p)^(-p) * p^(-dm)
    pen_m <- K_pen * ( sqrt(dm) + sqrt(2* log(1/pi_m)) )^2
    
    if (dm == 0) {
      fitted_vals <- rep(0, n)
    } else {
      X_sub <- X[, m, drop = FALSE]
      # Ajuste sin intercepto porque la base ya incluye la constante
      cf <- lm.fit(x = X_sub, y = Y)$coefficients
      fitted_vals <- X_sub %*% cf
    }
    rss <- sum((Y - fitted_vals)^2)
    return(rss + sigma^2 * pen_m)
  }
  
  # Empezamos con el modelo nulo
  m_actual <- integer(0)
  crit_actual <- criterio(m_actual)
  iter <- 0
  mejora <- TRUE
  
  while (mejora && iter < max_iter) {
    mejora <- FALSE
    iter <- iter + 1
    
    # Paso hacia adelante: buscar variable que más mejore el criterio
    candidatos_agregar <- setdiff(seq_len(p), m_actual)
    mejor_crit_agrega <- crit_actual
    var_a_agregar <- NA
    for (j in candidatos_agregar) {
      m_temp <- sort(c(m_actual, j))
      ctemp <- criterio(m_temp)
      if (ctemp < mejor_crit_agrega) {
        mejor_crit_agrega <- ctemp
        var_a_agregar <- j
      }
    }
    
    if (!is.na(var_a_agregar) && mejor_crit_agrega < crit_actual) {
      m_actual <- sort(c(m_actual, var_a_agregar))
      crit_actual <- mejor_crit_agrega
      mejora <- TRUE
    }
    
    # Paso hacia atrás: buscar variable cuya eliminación mejore el criterio
    if (length(m_actual) > 0) {
      candidatos_eliminar <- m_actual
      mejor_crit_elimina <- crit_actual
      var_a_eliminar <- NA
      for (j in candidatos_eliminar) {
        m_temp <- setdiff(m_actual, j)
        ctemp <- criterio(m_temp)
        if (ctemp < mejor_crit_elimina) {
          mejor_crit_elimina <- ctemp
          var_a_eliminar <- j
        }
      }
      if (!is.na(var_a_eliminar) && mejor_crit_elimina < crit_actual) {
        m_actual <- setdiff(m_actual, var_a_eliminar)
        crit_actual <- mejor_crit_elimina
        mejora <- TRUE
      }
    }
  } # fin while
  
  # Ajuste final
  if (length(m_actual) == 0) {
    f_hat <- rep(0, n)
  } else {
    X_sub <- X[, m_actual, drop = FALSE]
    cf <- lm.fit(x = X_sub, y = Y)$coefficients
    f_hat <- X_sub %*% cf
  }
  
  return(list(selected = sort(m_actual),
              fitted = f_hat,
              criterion = crit_actual,
              iterations = iter))
}

# --------------------------------------------------
# 5. Ajuste del modelo y gráfica
# --------------------------------------------------
K_pen <- 1.1  # Ajusta la constante de penalización a tu gusto
res <- backward_forward(X, Y, sigma = sigma, K_pen = K_pen, max_iter = 200)

# Se obtiene la señal estimada
f_hat <- as.vector(res$fitted)

# Graficamos
# - Puntos ruidosos: color gris
# - Señal real: línea discontinua
# - Señal estimada: línea roja
plot(
  x, Y, pch = 16, col = "gray50", 
  main = "Estimación de función con selección de modelos",
  xlab = "x", ylab = "y"
)
lines(x, ftrue(x), lty = 2, lwd = 2)   # señal real (línea discontinua)
lines(x, f_hat, col = "red", lwd = 2) # señal estimada en rojo
legend("topright", legend = c("Observaciones", "Señal real", "Señal estimada"),
       col = c("gray50", "black", "red"), lty = c(NA, 2, 1), pch = c(16, NA, NA),
       pt.cex = c(1, NA, NA), lwd = c(NA, 2, 2))












