# Función que implementa el algoritmo Backward-Forward
backward_forward <- function(X, Y, sigma, K = 1.1, max_iter = 100) {
  n <- length(Y)
  p <- ncol(X)
  
  # Función para calcular el criterio en un modelo dado (m: vector de índices de variables)
  criterio <- function(m) {
    dm <- length(m)
    # Calcular la penalización:
    # Observa que log(1/pi_m) = p*log(1+1/p) + dm * log(p)
    pen <- K * ( sqrt(dm) + sqrt(2*( p * log(1 + 1/p) + dm * log(p) ) ) )^2
    
    # Calcular el ajuste: si m es vacío, usamos f_hat = 0
    if (dm == 0) {
      fitted_vals <- rep(0, n)
    } else {
      X_sub <- X[, m, drop = FALSE]
      # Usamos lm.fit sin intercepto para obtener los coeficientes
      ajuste <- lm.fit(x = X_sub, y = Y)
      # Si el modelo presenta problemas numéricos, se puede usar solve() con ginv, 
      # pero para nuestros ejemplos asumimos que X_sub tiene rango completo.
      fitted_vals <- X_sub %*% ajuste$coefficients
    }
    
    rss <- sum((Y - fitted_vals)^2)
    return(rss + sigma^2 * pen)
  }
  
  # Inicialización del modelo: comenzamos con el modelo nulo
  m_actual <- integer(0)
  crit_actual <- criterio(m_actual)
  iter <- 0
  mejora <- TRUE
  
  while (mejora && iter < max_iter) {
    mejora <- FALSE
    iter <- iter + 1
    
    # -- Paso hacia adelante: buscar la variable que al agregarla disminuya el criterio --
    candidatos_agregar <- setdiff(1:p, m_actual)
    crit_min_agregar <- crit_actual
    var_a_agregar <- NA
    
    for (j in candidatos_agregar) {
      m_temp <- sort(c(m_actual, j))
      crit_temp <- criterio(m_temp)
      if (crit_temp < crit_min_agregar) {
        crit_min_agregar <- crit_temp
        var_a_agregar <- j
      }
    }
    
    if (!is.na(var_a_agregar) && (crit_min_agregar <= crit_actual)) {
      m_actual <- sort(c(m_actual, var_a_agregar))
      crit_actual <- crit_min_agregar
      mejora <- TRUE
      # Se puede imprimir el progreso:
      cat(sprintf("Iter %d - Adición: se añadió la variable %d. Nuevo criterio: %f\n", iter, var_a_agregar, crit_actual))
    }
    
    # -- Paso hacia atrás: intentar eliminar una variable que mejore el criterio --
    if (length(m_actual) > 0) {
      candidatos_eliminar <- m_actual
      crit_min_eliminar <- crit_actual
      var_a_eliminar <- NA
      
      for (j in candidatos_eliminar) {
        m_temp <- setdiff(m_actual, j)
        crit_temp <- criterio(m_temp)
        if (crit_temp < crit_min_eliminar) {
          crit_min_eliminar <- crit_temp
          var_a_eliminar <- j
        }
      }
      
      if (!is.na(var_a_eliminar) && (crit_min_eliminar < crit_actual)) {
        m_actual <- setdiff(m_actual, var_a_eliminar)
        crit_actual <- crit_min_eliminar
        mejora <- TRUE
        cat(sprintf("Iter %d - Eliminación: se eliminó la variable %d. Nuevo criterio: %f\n", iter, var_a_eliminar, crit_actual))
      }
    }
  }
  
  # Calcular el ajuste final (proyección de Y sobre S_m)
  if (length(m_actual) == 0) {
    f_hat <- rep(0, n)
  } else {
    X_sub <- X[, m_actual, drop = FALSE]
    ajuste_final <- lm.fit(x = X_sub, y = Y)
    f_hat <- X_sub %*% ajuste_final$coefficients
  }
  
  return(list(selected = sort(m_actual),
              fitted = f_hat,
              criterion = crit_actual,
              iterations = iter))
}


############################
########## Ejemplo 1 #######
############################

set.seed(123)
n <- 100; p <- 20
X <- matrix(rnorm(n * p), n, p)
beta <- rep(0, p)
beta[1:(p/2)] <- runif(p/2, min = -3, max = 3)  # m^* = {1, ..., 10}
cat("Beta (Ejemplo 1):", beta, "\n")
sigma <- 1
Y <- X %*% beta + rnorm(n, sd = sigma)

res1 <- backward_forward(X, Y, sigma, K = 1.1, max_iter = 100)
cat("Variables seleccionadas (Ejemplo 1):", res1$selected, "\n")

############################
########## Ejemplo 2 #######
############################

set.seed(1234)
n <- 100; p <- 20
X <- matrix(rnorm(n * p), n, p)
beta <- rep(0, p)
beta[1:(p/2)] <- rep(0.3, p/2)  # m^* = {1, ..., 10} pero con señal muy débil
cat("Beta (Ejemplo 2):", beta, "\n")
sigma <- 1
Y <- X %*% beta + rnorm(n, sd = sigma)


res_fail <- backward_forward(X, Y, sigma, K = 1.1, max_iter = 100)
cat("Variables seleccionadas en el escenario problemático:", res_fail$selected, "\n")