set.seed(12345)

# Definimos n y p: n relativamente grande y p aproximadamente 0.5*n o p menor a n.
n <- 2000
p <- 1000

# Generamos una matriz X aleatoria y luego la ortogonalizamos (columnas ortonormales)
X_raw <- matrix(rnorm(n * p), n, p)
X <- qr.Q(qr(X_raw))  # Ahora X es n x p con columnas ortogonales (y de norma 1)

# Fijamos sigma
sigma <- 1

# Definimos el vector beta: activamos los primeros 20 índices (por ejemplo) con un valor borderline (p.ej., 3.8)
beta <- rep(0, p)
active_indices <- 1:200         # m verdadero "global" (por fuerza bruta o umbral)
beta[active_indices] <- 3.8     # valor moderado; de forma individual es casi marginal

# Generamos la respuesta Y según el modelo: Y = X beta + ruido
Y <- X %*% beta + rnorm(n, sd = sigma)

# Elegimos el valor de K (constante de penalización)
K_val <- 1.1

# Calculamos lambda de acuerdo con la fórmula:
lambda <- K_val * ( 1 + sqrt(2 * log(p)) )^2
cat("Lambda =", lambda, "\n")

# --- Cálculo del estimador m mediante la regla de umbral (estimador tipo fuerza bruta) ---
# Debido a que las columnas de X están normalizadas, ||X_j||^2 = 1
# Se calcula el vector de correlaciones: X_j^T Y para cada j
correlations <- as.vector(t(X) %*% Y)
m_threshold <- which(correlations^2 > lambda)
cat("Variables seleccionadas por regla de umbral (m_threshold):", m_threshold, "\n")

# --- Ejecutamos el algoritmo greedy backward-forward ---
# Se asume que la función backward_forward ya está definida en el entorno.
res_greedy <- backward_forward(X, Y, sigma, K = K_val, max_iter = 100)
cat("Variables seleccionadas por backward_forward:", res_greedy$selected, "\n")

# --- intersección de variables en ambos métodos ---
intersect_vars <- intersect(m_threshold, res_greedy$selected)
cat("Intersección de variables entre ambos métodos:", intersect_vars, "\n")
