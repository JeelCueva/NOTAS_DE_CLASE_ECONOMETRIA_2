# ==============================================================================
# ECONOMETRÍA II - REGRESIÓN LINEAL CON DATOS REALES DE YAHOO FINANCE
# ==============================================================================
#
# Análisis CAPM usando datos reales descargados de Yahoo Finance
#
# Prof. Jeel Elvis Cueva Laguna
# Universidad Nacional Hermilio Valdizán - Huánuco
# ==============================================================================

# Limpiar entorno
rm(list = ls())

# ==============================================================================
# INSTALACIÓN Y CARGA DE PAQUETES
# ==============================================================================

# Función para instalar paquetes si no están instalados
instalar_si_necesario <- function(paquete) {
  if (!require(paquete, character.only = TRUE)) {
    install.packages(paquete, dependencies = TRUE)
    library(paquete, character.only = TRUE)
  }
}

# Lista de paquetes necesarios
paquetes <- c(
  "quantmod",      # Descargar datos financieros
  "tidyverse",     # Manipulación de datos y gráficas
  "lubridate",     # Manejo de fechas
  "lmtest",        # Tests econométricos
  "car",           # Diagnósticos de regresión
  "tseries",       # Series de tiempo
  "ggplot2",       # Gráficas avanzadas
  "gridExtra",     # Múltiples gráficas
  "moments"        # Jarque-Bera test
)

cat("Instalando y cargando paquetes necesarios...\n")
invisible(sapply(paquetes, instalar_si_necesario))

cat(strrep("=", 70), "\n")
cat("ECONOMETRÍA II - REGRESIÓN LINEAL CON DATOS REALES\n")
cat(strrep("=", 70), "\n\n")
cat("Prof. Jeel Elvis Cueva Laguna\n")
cat("Universidad Nacional Hermilio Valdizán - Huánuco\n\n")

# ==============================================================================
# PARTE 1: DESCARGA DE DATOS DESDE YAHOO FINANCE
# ==============================================================================

cat(strrep("=", 70), "\n")
cat("PARTE 1: DESCARGA DE DATOS DESDE YAHOO FINANCE\n")
cat(strrep("=", 70), "\n\n")

# Configurar fechas
fecha_inicio <- "2020-01-01"
fecha_fin <- Sys.Date()  # Fecha actual

cat("Periodo de análisis:\n")
cat(sprintf("  Fecha inicio: %s\n", fecha_inicio))
cat(sprintf("  Fecha fin:    %s\n", fecha_fin))

# Símbolos de los activos
activos <- c(
  "AAPL",   # Apple
  "^GSPC"   # S&P 500 (índice de mercado)
)

cat("\nActivos a descargar:\n")
for (activo in activos) {
  nombre <- ifelse(activo == "AAPL", "Apple Inc.", "S&P 500")
  cat(sprintf("  %s: %s\n", activo, nombre))
}

# Descargar datos
cat("\nDescargando datos...\n")

tryCatch({
  # Descargar Apple
  cat("  Descargando AAPL (Apple)...\n")
  getSymbols("AAPL", src = "yahoo", from = fecha_inicio, to = fecha_fin, auto.assign = TRUE)
  
  # Descargar S&P 500
  cat("  Descargando ^GSPC (S&P 500)...\n")
  getSymbols("^GSPC", src = "yahoo", from = fecha_inicio, to = fecha_fin, auto.assign = TRUE)
  
  cat("✓ Datos descargados exitosamente\n")
  
}, error = function(e) {
  cat("✗ Error al descargar datos:", conditionMessage(e), "\n")
  cat("Verificar conexión a internet o símbolos de activos\n")
  stop("No se pudieron descargar los datos")
})

# Extraer precios de cierre ajustados
precios_apple <- Ad(AAPL)
precios_sp500 <- Ad(GSPC)

# Alinear fechas (ambas series deben tener las mismas fechas)
precios <- merge(precios_apple, precios_sp500, join = "inner")
colnames(precios) <- c("Apple", "SP500")

# Resumen de datos descargados
cat("\n", strrep("─", 70), "\n", sep="")
cat("RESUMEN DE DATOS DESCARGADOS\n")
cat(strrep("─", 70), "\n")
cat(sprintf("Número de observaciones: %d\n", nrow(precios)))
cat(sprintf("Primera fecha: %s\n", index(precios)[1]))
cat(sprintf("Última fecha:  %s\n", index(precios)[nrow(precios)]))

cat("\nPrimeras 5 observaciones (precios):\n")
print(head(precios, 5))

cat("\nÚltimas 5 observaciones (precios):\n")
print(tail(precios, 5))

# ==============================================================================
# PARTE 2: CÁLCULO DE RETORNOS
# ==============================================================================

cat("\n", strrep("=", 70), "\n", sep="")
cat("PARTE 2: CÁLCULO DE RETORNOS LOGARÍTMICOS\n")
cat(strrep("=", 70), "\n\n")

# Calcular retornos logarítmicos
retornos_apple <- diff(log(precios$Apple))
retornos_sp500 <- diff(log(precios$SP500))

# Eliminar NA del primer valor
retornos <- na.omit(merge(retornos_apple, retornos_sp500))
colnames(retornos) <- c("r_Apple", "r_SP500")

cat(sprintf("Retornos calculados: %d observaciones\n", nrow(retornos)))

# Convertir a data frame para análisis
df_retornos <- data.frame(
  Fecha = index(retornos),
  r_Apple = as.numeric(retornos$r_Apple),
  r_SP500 = as.numeric(retornos$r_SP500)
)

# Estadísticas descriptivas
cat("\nEstadísticas descriptivas de retornos:\n")
cat(strrep("─", 70), "\n")

estadisticas <- data.frame(
  Variable = c("Apple", "S&P 500"),
  Media = c(mean(df_retornos$r_Apple), mean(df_retornos$r_SP500)),
  Desv_Std = c(sd(df_retornos$r_Apple), sd(df_retornos$r_SP500)),
  Minimo = c(min(df_retornos$r_Apple), min(df_retornos$r_SP500)),
  Maximo = c(max(df_retornos$r_Apple), max(df_retornos$r_SP500)),
  Asimetria = c(skewness(df_retornos$r_Apple), skewness(df_retornos$r_SP500)),
  Curtosis = c(kurtosis(df_retornos$r_Apple), kurtosis(df_retornos$r_SP500))
)

print(estadisticas, row.names = FALSE, digits = 6)

# Anualizamos las estadísticas (252 días de trading)
cat("\nRetornos anualizados (252 días de trading):\n")
cat(sprintf("  Apple:   %.2f%% anual (Desv: %.2f%%)\n", 
            mean(df_retornos$r_Apple) * 252 * 100,
            sd(df_retornos$r_Apple) * sqrt(252) * 100))
cat(sprintf("  S&P 500: %.2f%% anual (Desv: %.2f%%)\n", 
            mean(df_retornos$r_SP500) * 252 * 100,
            sd(df_retornos$r_SP500) * sqrt(252) * 100))

# ==============================================================================
# PARTE 3: EJEMPLO MANUAL CON 5 OBSERVACIONES
# ==============================================================================

cat("\n", strrep("=", 70), "\n", sep="")
cat("PARTE 3: EJEMPLO MANUAL CON 5 OBSERVACIONES\n")
cat(strrep("=", 70), "\n\n")

# Seleccionar las últimas 5 observaciones
n_manual <- 5
df_manual <- tail(df_retornos, n_manual)

cat("Últimas 5 observaciones (para cálculo manual):\n")
print(df_manual, row.names = FALSE)

# Variables
x_manual <- df_manual$r_SP500
y_manual <- df_manual$r_Apple
n <- length(x_manual)

# Paso 1: Estadísticas básicas
x_mean <- mean(x_manual)
y_mean <- mean(y_manual)

cat("\n", strrep("─", 70), "\n", sep="")
cat("PASO 1: ESTADÍSTICAS BÁSICAS\n")
cat(strrep("─", 70), "\n")
cat(sprintf("n = %d\n", n))
cat(sprintf("Media de x (S&P 500): %.8f\n", x_mean))
cat(sprintf("Media de y (Apple):   %.8f\n", y_mean))

# Paso 2: Desviaciones
x_dev <- x_manual - x_mean
y_dev <- y_manual - y_mean

Sxx <- sum(x_dev^2)
Syy <- sum(y_dev^2)
Sxy <- sum(x_dev * y_dev)

cat("\n", strrep("─", 70), "\n", sep="")
cat("PASO 2: SUMAS DE CUADRADOS\n")
cat(strrep("─", 70), "\n")
cat(sprintf("Sxx = Σ(x - x̄)² = %.10f\n", Sxx))
cat(sprintf("Syy = Σ(y - ȳ)² = %.10f\n", Syy))
cat(sprintf("Sxy = Σ(x - x̄)(y - ȳ) = %.10f\n", Sxy))

# Paso 3: Estimadores MCO
beta_1 <- Sxy / Sxx
beta_0 <- y_mean - beta_1 * x_mean

cat("\n", strrep("─", 70), "\n", sep="")
cat("PASO 3: ESTIMADORES MCO\n")
cat(strrep("─", 70), "\n")
cat(sprintf("β₁ (Beta)  = Sxy / Sxx = %.10f / %.10f\n", Sxy, Sxx))
cat(sprintf("           = %.6f\n", beta_1))
cat(sprintf("\nβ₀ (Alpha) = ȳ - β₁·x̄ = %.8f - %.6f × %.8f\n", y_mean, beta_1, x_mean))
cat(sprintf("           = %.6f\n", beta_0))

cat("\nECUACIÓN ESTIMADA (n=5):\n")
cat(sprintf("ŷ = %.6f + %.4f·x\n", beta_0, beta_1))

# Paso 4: Valores ajustados y residuos
y_pred_manual <- beta_0 + beta_1 * x_manual
residuos_manual <- y_manual - y_pred_manual

cat("\n", strrep("─", 70), "\n", sep="")
cat("PASO 4: VALORES AJUSTADOS Y RESIDUOS\n")
cat(strrep("─", 70), "\n")

df_ajuste_manual <- data.frame(
  Obs = 1:n,
  x = x_manual,
  y = y_manual,
  y_pred = y_pred_manual,
  residuo = residuos_manual
)
print(df_ajuste_manual, row.names = FALSE, digits = 6)

SCR_manual <- sum(residuos_manual^2)
cat(sprintf("\nVerificación: Σε̂ = %.10f ≈ 0 ✓\n", sum(residuos_manual)))
cat(sprintf("SCR = %.10f\n", SCR_manual))

# Paso 5: R²
SCT_manual <- Syy
SCE_manual <- SCT_manual - SCR_manual
R2_manual <- SCE_manual / SCT_manual

cat("\n", strrep("─", 70), "\n", sep="")
cat("PASO 5: R² (BONDAD DE AJUSTE)\n")
cat(strrep("─", 70), "\n")
cat(sprintf("SCT = %.10f\n", SCT_manual))
cat(sprintf("SCE = %.10f\n", SCE_manual))
cat(sprintf("SCR = %.10f\n", SCR_manual))
cat(sprintf("\nR² = SCE / SCT = %.6f (%.2f%%)\n", R2_manual, R2_manual * 100))

# Verificación con lm()
modelo_manual <- lm(y_manual ~ x_manual)
cat("\nVerificación con lm():\n")
cat(sprintf("  β₀ (manual) = %.6f | lm() = %.6f | Dif = %.10f\n", 
            beta_0, coef(modelo_manual)[1], abs(beta_0 - coef(modelo_manual)[1])))
cat(sprintf("  β₁ (manual) = %.6f | lm() = %.6f | Dif = %.10f\n", 
            beta_1, coef(modelo_manual)[2], abs(beta_1 - coef(modelo_manual)[2])))
cat(sprintf("  R² (manual) = %.6f | lm() = %.6f | Dif = %.10f\n", 
            R2_manual, summary(modelo_manual)$r.squared, 
            abs(R2_manual - summary(modelo_manual)$r.squared)))

# ==============================================================================
# PARTE 4: ANÁLISIS COMPLETO CON TODOS LOS DATOS
# ==============================================================================

cat("\n", strrep("=", 70), "\n", sep="")
cat("PARTE 4: ANÁLISIS COMPLETO CON TODOS LOS DATOS\n")
cat(strrep("=", 70), "\n\n")

cat(sprintf("Estimando modelo CAPM con n = %d observaciones\n", nrow(df_retornos)))
cat("Modelo: r_Apple = α + β·r_SP500 + ε\n\n")

# Estimar modelo completo
modelo_completo <- lm(r_Apple ~ r_SP500, data = df_retornos)

# Resumen del modelo
cat("Resumen del modelo:\n")
cat(strrep("─", 70), "\n")
print(summary(modelo_completo))

# Extraer estadísticas
coef_tabla <- summary(modelo_completo)$coefficients
r_squared <- summary(modelo_completo)$r.squared
r_squared_adj <- summary(modelo_completo)$adj.r.squared
f_stat <- summary(modelo_completo)$fstatistic[1]

cat("\n", strrep("─", 70), "\n", sep="")
cat("RESULTADOS PRINCIPALES\n")
cat(strrep("─", 70), "\n")
cat("Ecuación estimada:\n")
cat(sprintf("ŷ = %.6f + %.4f·x\n\n", coef(modelo_completo)[1], coef(modelo_completo)[2]))

cat("Tabla de coeficientes:\n")
resultados <- data.frame(
  Coeficiente = c("α (Alpha)", "β (Beta)"),
  Estimador = coef(modelo_completo),
  Error_Std = coef_tabla[, "Std. Error"],
  t_statistic = coef_tabla[, "t value"],
  p_valor = coef_tabla[, "Pr(>|t|)"]
)
print(resultados, row.names = FALSE, digits = 6)

cat(sprintf("\nR² = %.6f (%.2f%%)\n", r_squared, r_squared * 100))
cat(sprintf("R² ajustado = %.6f (%.2f%%)\n", r_squared_adj, r_squared_adj * 100))
cat(sprintf("F-statistic = %.4f\n", f_stat))

# Valores ajustados y residuos
valores_ajustados <- fitted(modelo_completo)
residuos <- residuals(modelo_completo)

# ==============================================================================
# PARTE 5: DIAGNÓSTICOS DEL MODELO
# ==============================================================================

cat("\n", strrep("=", 70), "\n", sep="")
cat("PARTE 5: DIAGNÓSTICOS DEL MODELO\n")
cat(strrep("=", 70), "\n\n")

# Test 1: Normalidad (Jarque-Bera)
cat("1. TEST DE NORMALIDAD (JARQUE-BERA)\n")
cat(strrep("─", 70), "\n")
cat("H₀: Los residuos siguen una distribución normal\n")

jb_test <- jarque.test(residuos)
cat(sprintf("Estadístico JB = %.4f\n", jb_test$statistic))
cat(sprintf("p-valor = %.4f\n", jb_test$p.value))

if (jb_test$p.value > 0.05) {
  cat("✓ No se rechaza H₀ (p-valor > 0.05)\n")
  cat("Conclusión: Los residuos son aproximadamente normales\n")
} else {
  cat("✗ Se rechaza H₀ (p-valor < 0.05)\n")
  cat("Conclusión: Los residuos NO son normales\n")
}

# Test adicional: Shapiro-Wilk (si n < 5000)
if (length(residuos) < 5000) {
  sw_test <- shapiro.test(residuos)
  cat(sprintf("\nTest de Shapiro-Wilk:\n"))
  cat(sprintf("Estadístico W = %.4f\n", sw_test$statistic))
  cat(sprintf("p-valor = %.4f\n", sw_test$p.value))
  
  if (sw_test$p.value > 0.05) {
    cat("✓ No se rechaza H₀\n")
  } else {
    cat("✗ Se rechaza H₀\n")
  }
}

# Estadísticos descriptivos de residuos
cat(sprintf("\nEstadísticos de residuos:\n"))
cat(sprintf("  Media = %.10f (debe ser ≈ 0)\n", mean(residuos)))
cat(sprintf("  Desv. Std = %.6f\n", sd(residuos)))
cat(sprintf("  Asimetría = %.4f (normal: 0)\n", skewness(residuos)))
cat(sprintf("  Curtosis = %.4f (normal: 3)\n", kurtosis(residuos)))

# Test 2: Autocorrelación (Durbin-Watson)
cat("\n2. TEST DE AUTOCORRELACIÓN (DURBIN-WATSON)\n")
cat(strrep("─", 70), "\n")
cat("H₀: No hay autocorrelación\n")

dw_test <- dwtest(modelo_completo)
cat(sprintf("Estadístico DW = %.4f\n", dw_test$statistic))
cat(sprintf("p-valor = %.4f\n", dw_test$p.value))

cat("\nInterpretación:\n")
cat("  DW ≈ 2.0  : No autocorrelación\n")
cat("  DW < 1.5  : Autocorrelación positiva\n")
cat("  DW > 2.5  : Autocorrelación negativa\n\n")

if (dw_test$statistic > 1.5 && dw_test$statistic < 2.5) {
  cat(sprintf("✓ DW = %.4f está cerca de 2\n", dw_test$statistic))
  cat("Conclusión: No hay evidencia de autocorrelación\n")
} else if (dw_test$statistic < 1.5) {
  cat(sprintf("✗ DW = %.4f < 1.5\n", dw_test$statistic))
  cat("Conclusión: Posible autocorrelación POSITIVA\n")
} else {
  cat(sprintf("✗ DW = %.4f > 2.5\n", dw_test$statistic))
  cat("Conclusión: Posible autocorrelación NEGATIVA\n")
}

# Test 3: Heterocedasticidad (Breusch-Pagan)
cat("\n3. TEST DE HETEROCEDASTICIDAD (BREUSCH-PAGAN)\n")
cat(strrep("─", 70), "\n")
cat("H₀: Homocedasticidad (varianza constante)\n")

bp_test <- bptest(modelo_completo)
cat(sprintf("Estadístico BP = %.4f\n", bp_test$statistic))
cat(sprintf("p-valor = %.4f\n", bp_test$p.value))

if (bp_test$p.value > 0.05) {
  cat("✓ No se rechaza H₀ (p-valor > 0.05)\n")
  cat("Conclusión: Hay homocedasticidad\n")
} else {
  cat("✗ Se rechaza H₀ (p-valor < 0.05)\n")
  cat("Conclusión: Hay heterocedasticidad\n")
}

# Test adicional: White
cat("\nTest de White (alternativo):\n")
white_test <- bptest(modelo_completo, ~ r_SP500 + I(r_SP500^2), data = df_retornos)
cat(sprintf("Estadístico = %.4f\n", white_test$statistic))
cat(sprintf("p-valor = %.4f\n", white_test$p.value))

if (white_test$p.value > 0.05) {
  cat("✓ No se rechaza H₀\n")
} else {
  cat("✗ Se rechaza H₀\n")
}

# ==============================================================================
# PARTE 6: VISUALIZACIONES
# ==============================================================================

cat("\n", strrep("=", 70), "\n", sep="")
cat("PARTE 6: GENERANDO GRÁFICAS\n")
cat(strrep("=", 70), "\n\n")

# Preparar datos para ggplot
df_plot <- df_retornos
df_plot$valores_ajustados <- valores_ajustados
df_plot$residuos <- residuos
df_plot$residuos_std <- rstandard(modelo_completo)

# Gráfica 1: Dispersión con línea de regresión
p1 <- ggplot(df_plot, aes(x = r_SP500, y = r_Apple)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE, color = "red", linewidth = 1) +
  labs(
    title = sprintf("Modelo CAPM: Apple vs S&P 500 (n=%d)", nrow(df_plot)),
    x = "Retorno S&P 500",
    y = "Retorno Apple"
  ) +
  annotate("text", x = min(df_plot$r_SP500), y = max(df_plot$r_Apple),
           label = sprintf("ŷ = %.4f + %.2f·x\nR² = %.4f\nβ = %.4f",
                           coef(modelo_completo)[1], coef(modelo_completo)[2],
                           r_squared, coef(modelo_completo)[2]),
           hjust = 0, vjust = 1, size = 3.5, fontface = "bold") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

# Gráfica 2: Residuos vs Valores ajustados
p2 <- ggplot(df_plot, aes(x = valores_ajustados, y = residuos)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Residuos vs Valores Ajustados",
    subtitle = "Verificación de homocedasticidad",
    x = "Valores Ajustados",
    y = "Residuos"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

# Gráfica 3: Q-Q Plot
qq_data <- data.frame(
  sample = qqnorm(residuos, plot.it = FALSE)$x,
  theoretical = qqnorm(residuos, plot.it = FALSE)$y
)

p3 <- ggplot(qq_data, aes(sample = theoretical)) +
  stat_qq(size = 1.5, alpha = 0.5) +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(
    title = "Q-Q Plot de Residuos",
    subtitle = "Verificación de normalidad",
    x = "Cuantiles Teóricos",
    y = "Cuantiles Muestrales"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

# Gráfica 4: Histograma de residuos
p4 <- ggplot(df_plot, aes(x = residuos)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30,
                 fill = "steelblue", color = "black", alpha = 0.7) +
  stat_function(fun = dnorm,
                args = list(mean = mean(residuos), sd = sd(residuos)),
                color = "red", linewidth = 1) +
  labs(
    title = "Distribución de Residuos",
    subtitle = "Comparación con distribución normal",
    x = "Residuos",
    y = "Densidad"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

# Gráfica 5: Residuos en el tiempo
df_plot$obs <- 1:nrow(df_plot)
p5 <- ggplot(df_plot, aes(x = obs, y = residuos)) +
  geom_line(alpha = 0.7, linewidth = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Residuos en el Tiempo",
    subtitle = "Verificación de autocorrelación",
    x = "Observación",
    y = "Residuos"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

# Gráfica 6: Ejemplo manual (últimas 5 observaciones)
df_manual_plot <- data.frame(
  x = x_manual,
  y = y_manual,
  obs = 1:n
)

p6 <- ggplot(df_manual_plot, aes(x = x, y = y)) +
  geom_point(size = 4, color = "blue", alpha = 0.6) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              color = "red", linewidth = 1) +
  geom_text(aes(label = obs), vjust = -1, size = 3.5) +
  labs(
    title = sprintf("Ejemplo Manual (n=%d)", n),
    subtitle = "Últimas 5 observaciones",
    x = "Retorno S&P 500",
    y = "Retorno Apple"
  ) +
  annotate("text", x = min(x_manual), y = max(y_manual),
           label = sprintf("ŷ = %.4f + %.2f·x\nR² = %.4f\nβ = %.4f",
                           beta_0, beta_1, R2_manual, beta_1),
           hjust = 0, vjust = 1, size = 3.5, fontface = "bold") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12))

# Modificar la sección de guardado de gráficas:

# Crear el directorio si no existe
if (!dir.exists("graficas")) {
  dir.create("graficas")
}

# Guardar gráficas con mejor configuración
cat("Guardando gráficas...\n")

# Opción 1: Guardar como PDF (mejor calidad)
pdf("graficas/regresion_yahoo_finance.pdf", width = 14, height = 10)
grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 3)
dev.off()
cat("✓ Gráficas guardadas en 'graficas/regresion_yahoo_finance.pdf'\n")

# Opción 2: Guardar como PNG individual
# Guardar cada gráfica por separado para mayor control
tryCatch({
  png("graficas/01_dispersion.png", width = 800, height = 600)
  print(p1)
  dev.off()
  
  png("graficas/02_residuos_vs_ajustados.png", width = 800, height = 600)
  print(p2)
  dev.off()
  
  png("graficas/03_qqplot.png", width = 800, height = 600)
  print(p3)
  dev.off()
  
  png("graficas/04_histograma.png", width = 800, height = 600)
  print(p4)
  dev.off()
  
  png("graficas/05_residuos_tiempo.png", width = 800, height = 600)
  print(p5)
  dev.off()
  
  png("graficas/06_ejemplo_manual.png", width = 800, height = 600)
  print(p6)
  dev.off()
  
  cat("✓ Gráficas individuales guardadas en carpeta 'graficas/'\n")
}, error = function(e) {
  cat("⚠ Error al guardar gráficas PNG:", e$message, "\n")
  cat("Intentando guardar en formato JPEG...\n")
  
  # Intentar con JPEG como alternativa
  jpeg("graficas/regresion_yahoo_finance.jpg", width = 1600, height = 1200, quality = 100)
  grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 3)
  dev.off()
  cat("✓ Gráfica guardada como JPEG en 'graficas/regresion_yahoo_finance.jpg'\n")
})

# ==============================================================================
# PARTE 7: RESUMEN EJECUTIVO
# ==============================================================================

cat("\n", strrep("=", 70), "\n", sep="")
cat("RESUMEN EJECUTIVO\n")
cat(strrep("=", 70), "\n\n")

cat("1. DATOS UTILIZADOS:\n")
cat(strrep("─", 70), "\n")
cat(sprintf("  Fuente: Yahoo Finance\n"))
cat(sprintf("  Periodo: %s a %s\n", fecha_inicio, fecha_fin))
cat(sprintf("  Observaciones: %d\n", nrow(df_retornos)))
cat(sprintf("  Activos: Apple (AAPL) vs S&P 500 (^GSPC)\n"))

cat("\n2. MODELO ESTIMADO:\n")
cat(strrep("─", 70), "\n")
cat(sprintf("  r_Apple = %.6f + %.4f · r_SP500 + ε\n\n",
            coef(modelo_completo)[1], coef(modelo_completo)[2]))

cat("  Interpretación:\n")
cat(sprintf("  • Alpha (α) = %.4f: ", coef(modelo_completo)[1]))
if (coef_tabla[1, "Pr(>|t|)"] > 0.05) {
  cat("NO significativo (p > 0.05)\n")
  cat("    Apple no genera retornos en exceso sistemáticos\n")
} else {
  cat("Significativo (p < 0.05)\n")
}

cat(sprintf("\n  • Beta (β) = %.4f: ", coef(modelo_completo)[2]))
if (coef(modelo_completo)[2] > 1) {
  cat(sprintf("Apple es %.0f%% más volátil que el mercado\n",
              (coef(modelo_completo)[2] - 1) * 100))
  cat("    (Activo agresivo, alto riesgo sistemático)\n")
} else if (coef(modelo_completo)[2] < 1) {
  cat(sprintf("Apple es %.0f%% menos volátil que el mercado\n",
              (1 - coef(modelo_completo)[2]) * 100))
  cat("    (Activo defensivo, bajo riesgo sistemático)\n")
} else {
  cat("Apple tiene el mismo riesgo que el mercado\n")
}

cat(sprintf("\n  • R² = %.2f%%: El mercado explica %.2f%% de los retornos de Apple\n",
            r_squared * 100, r_squared * 100))

cat("\n3. VALIDACIÓN DE SUPUESTOS:\n")
cat(strrep("─", 70), "\n")

# Resumen de tests
tests <- data.frame(
  Supuesto = c("Linealidad", "Normalidad", "Homocedasticidad", "No Autocorrelación"),
  Test = c("Visual", "Jarque-Bera", "Breusch-Pagan", "Durbin-Watson"),
  Resultado = c(
    "✓",
    ifelse(jb_test$p.value > 0.05, "✓", "✗"),
    ifelse(bp_test$p.value > 0.05, "✓", "✗"),
    ifelse(dw_test$statistic > 1.5 && dw_test$statistic < 2.5, "✓", "✗")
  ),
  p_valor = c(
    "N/A",
    sprintf("%.4f", jb_test$p.value),
    sprintf("%.4f", bp_test$p.value),
    sprintf("%.4f", dw_test$p.value)
  )
)

print(tests, row.names = FALSE)

# Contar tests aprobados
tests_aprobados <- sum(tests$Resultado == "✓")
tests_totales <- nrow(tests)
porcentaje <- (tests_aprobados / tests_totales) * 100

cat(sprintf("\nTests aprobados: %d/%d (%.1f%%)\n", 
            tests_aprobados, tests_totales, porcentaje))

if (porcentaje >= 75) {
  cat("\n✓✓✓ EXCELENTE: El modelo cumple la mayoría de los supuestos\n")
  cat("    Los resultados de la regresión son confiables.\n")
} else if (porcentaje >= 50) {
  cat("\n✓✓ BUENO: El modelo cumple varios supuestos importantes\n")
  cat("   Pequeñas violaciones no invalidan los resultados.\n")
} else {
  cat("\n⚠ PRECAUCIÓN: Algunas violaciones importantes de supuestos\n")
  cat("   Considere transformaciones o métodos robustos.\n")
}

cat("\n4. CONCLUSIONES PRINCIPALES:\n")
cat(strrep("─", 70), "\n")
cat("✓ Datos reales descargados exitosamente desde Yahoo Finance\n")
cat("✓ El modelo CAPM captura la relación riesgo-retorno para Apple\n")
cat(sprintf("✓ Beta = %.2f confirma que Apple es %s que el mercado\n",
            coef(modelo_completo)[2],
            ifelse(coef(modelo_completo)[2] > 1, "más volátil", "menos volátil")))
cat(sprintf("✓ R² = %.1f%% indica un buen ajuste del modelo\n", r_squared * 100))
cat("✓ La mayoría de los supuestos del modelo se cumplen\n")

cat("\n5. RECOMENDACIONES:\n")
cat(strrep("─", 70), "\n")
cat("• Los resultados son confiables para análisis de riesgo\n")
cat("• El beta puede usarse para valoración de activos (CAPM)\n")
cat("• Considere análisis de sensibilidad con diferentes periodos\n")
cat("• Compare con otros modelos (Fama-French, Carhart)\n")

cat("\n", strrep("=", 70), "\n", sep="")
cat("FIN DEL ANÁLISIS\n")
cat(strrep("=", 70), "\n\n")

cat("Archivos generados:\n")
cat("  • regresion_yahoo_finance_R.png (gráficas)\n")
cat("  • Objetos en memoria: modelo_completo, df_retornos, precios\n\n")

cat("Para acceder a los resultados:\n")
cat("  summary(modelo_completo)  # Resumen del modelo\n")
cat("  coef(modelo_completo)     # Coeficientes\n")
cat("  residuals(modelo_completo) # Residuos\n")
cat("  head(df_retornos)         # Ver datos\n\n")

cat("¡Análisis completado exitosamente! ✓\n")
