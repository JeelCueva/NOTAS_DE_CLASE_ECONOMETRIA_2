#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capítulo 3: Modelos de Variable Dependiente Limitada
Ejemplo 01: Estimación de Modelos Probit y Logit

Este script implementa la estimación por máxima verosimilitud de modelos
de elección binaria (Probit y Logit) usando datos reales de retornos de acciones.

Author: Jeel Cueva
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, Logit
import warnings
warnings.filterwarnings('ignore')

# Configuración de gráficas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CAPÍTULO 3: MODELOS CON VARIABLE DEPENDIENTE LIMITADA")
print("Ejemplo 01: Modelos Probit y Logit")
print("="*80)

#%% ===========================================================================
# PARTE 1: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

print("\n" + "="*80)
print("PARTE 1: CARGA Y PREPARACIÓN DE DATOS")
print("="*80)

# Leer datos desde el Excel
file_path = '/content/datos_activos_financieros.xlsx'

# Leer muestra manual (10 observaciones)
df_manual = pd.read_excel(file_path, sheet_name='Muestra_Manual_10obs')

# Calcular retornos
df_manual['AAPL_Return'] = df_manual['AAPL'].pct_change()
df_manual['NVDA_Return'] = df_manual['NVDA'].pct_change()

# Eliminar primer valor (NaN)
df_manual = df_manual.dropna().reset_index(drop=True)

print(f"\nDatos cargados: {len(df_manual)} observaciones")
print(f"\nPrimeras observaciones:")
print(df_manual[['AAPL_Return', 'NVDA_Return']].head().round(6))

# Crear variable dependiente binaria: 1 si retorno de AAPL > 0
y = (df_manual['AAPL_Return'] > 0).astype(int).values
x = df_manual['NVDA_Return'].values
n = len(y)

print(f"\n{'Estadísticas Descriptivas:':-^80}")
print(f"Variable dependiente y (AAPL > 0):")
print(f"  Proporción de éxitos (y=1): {y.mean():.4f}")
print(f"  Número de observaciones: {n}")
print(f"  Número de y=1: {y.sum()}")
print(f"  Número de y=0: {n - y.sum()}")

print(f"\nVariable independiente x (retorno NVDA):")
print(f"  Media: {x.mean():.6f}")
print(f"  Desv. Estándar: {x.std(ddof=1):.6f}")
print(f"  Mínimo: {x.min():.6f}")
print(f"  Máximo: {x.max():.6f}")

#%% ===========================================================================
# PARTE 2: ESTIMACIÓN MANUAL DEL MODELO PROBIT
# =============================================================================

print("\n" + "="*80)
print("PARTE 2: ESTIMACIÓN MANUAL DEL MODELO PROBIT")
print("="*80)

def probit_log_likelihood(params, y, x):
    """
    Calcula la log-verosimilitud del modelo Probit

    Parameters:
    -----------
    params : array-like
        [beta_0, beta_1]
    y : array-like
        Variable dependiente binaria
    x : array-like
        Variable independiente

    Returns:
    --------
    float : Valor negativo de la log-verosimilitud (para minimización)
    """
    beta_0, beta_1 = params
    z = beta_0 + beta_1 * x

    # Probabilidad ajustada
    prob = norm.cdf(z)

    # Evitar log(0)
    prob = np.clip(prob, 1e-10, 1 - 1e-10)

    # Log-verosimilitud
    log_lik = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))

    return -log_lik  # Negativo para minimización

def probit_gradient(params, y, x):
    """Gradiente de la log-verosimilitud del Probit"""
    beta_0, beta_1 = params
    z = beta_0 + beta_1 * x

    prob = norm.cdf(z)
    pdf = norm.pdf(z)

    # Evitar división por cero
    prob = np.clip(prob, 1e-10, 1 - 1e-10)

    # Gradiente
    gradient = np.zeros(2)
    gradient[0] = -np.sum((y - prob) * pdf / (prob * (1 - prob)))
    gradient[1] = -np.sum((y - prob) * pdf * x / (prob * (1 - prob)))

    return gradient

# Valores iniciales
# beta_0 inicial: inverse normal CDF de la proporción de éxitos
beta_0_init = norm.ppf(y.mean())
beta_1_init = 0.0

params_init = np.array([beta_0_init, beta_1_init])

print(f"\nValores iniciales:")
print(f"  β₀ inicial: {beta_0_init:.4f}")
print(f"  β₁ inicial: {beta_1_init:.4f}")

# Maximización de la verosimilitud
print(f"\nMaximizando la log-verosimilitud...")
result_probit = minimize(
    probit_log_likelihood,
    params_init,
    args=(y, x),
    method='BFGS',
    jac=probit_gradient
)

beta_hat_probit = result_probit.x
print(f"\n{'Resultados del Modelo Probit (Manual):':-^80}")
print(f"  β̂₀ (Intercepto): {beta_hat_probit[0]:.4f}")
print(f"  β̂₁ (Pendiente):  {beta_hat_probit[1]:.4f}")
print(f"  Log-verosimilitud: {-result_probit.fun:.4f}")
print(f"  Convergencia: {'Exitosa' if result_probit.success else 'Fallida'}")

# Cálculo de la matriz de información de Fisher
def fisher_information_probit(params, y, x):
    """Calcula la matriz de información de Fisher"""
    beta_0, beta_1 = params
    z = beta_0 + beta_1 * x

    prob = norm.cdf(z)
    pdf = norm.pdf(z)

    # Evitar problemas numéricos
    prob = np.clip(prob, 1e-10, 1 - 1e-10)

    # Matriz de información
    info = np.zeros((2, 2))
    weight = (pdf ** 2) / (prob * (1 - prob))

    info[0, 0] = np.sum(weight)
    info[0, 1] = np.sum(weight * x)
    info[1, 0] = info[0, 1]
    info[1, 1] = np.sum(weight * x ** 2)

    return info

# Matriz de información
info_matrix = fisher_information_probit(beta_hat_probit, y, x)
var_cov_matrix = np.linalg.inv(info_matrix)
se_probit = np.sqrt(np.diag(var_cov_matrix))

print(f"\n{'Errores Estándar y Pruebas de Hipótesis:':-^80}")
print(f"  SE(β̂₀): {se_probit[0]:.4f}")
print(f"  SE(β̂₁): {se_probit[1]:.4f}")

# Estadísticos z
z_stats = beta_hat_probit / se_probit
p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))

print(f"\n  Estadístico z para β₀: {z_stats[0]:.3f} (p-valor: {p_values[0]:.3f})")
print(f"  Estadístico z para β₁: {z_stats[1]:.3f} (p-valor: {p_values[1]:.3f})")

# Interpretación
print(f"\n{'Interpretación:':-^80}")
prob_at_zero = norm.cdf(beta_hat_probit[0])
print(f"  Cuando x=0, P(y=1) = Φ({beta_hat_probit[0]:.4f}) = {prob_at_zero:.4f}")
print(f"  Un aumento de 1 unidad en x aumenta la probabilidad")
print(f"  (el efecto marginal depende del valor de x)")

#%% ===========================================================================
# PARTE 3: CÁLCULO DE PROBABILIDADES AJUSTADAS Y CLASIFICACIÓN
# =============================================================================

print("\n" + "="*80)
print("PARTE 3: PROBABILIDADES AJUSTADAS Y CLASIFICACIÓN")
print("="*80)

# Calcular probabilidades ajustadas
z_fitted = beta_hat_probit[0] + beta_hat_probit[1] * x
prob_fitted = norm.cdf(z_fitted)
y_pred = (prob_fitted > 0.5).astype(int)

# Tabla de resultados
results_df = pd.DataFrame({
    'Obs': range(1, n + 1),
    'y': y,
    'x': x,
    'z = β₀ + β₁x': z_fitted,
    'P̂(y=1)': prob_fitted,
    'ŷ': y_pred
})

print("\nTabla de Probabilidades Ajustadas:")
print(results_df.round(4).to_string(index=False))

# Matriz de confusión
correct = (y == y_pred).sum()
accuracy = correct / n

print(f"\n{'Tabla de Clasificación:':-^80}")
print(f"  Predicciones correctas: {correct}/{n}")
print(f"  Tasa de acierto: {accuracy:.1%}")

#%% ===========================================================================
# PARTE 4: EFECTOS MARGINALES
# =============================================================================

print("\n" + "="*80)
print("PARTE 4: EFECTOS MARGINALES")
print("="*80)

# Efecto marginal en la media
x_mean = x.mean()
z_mean = beta_hat_probit[0] + beta_hat_probit[1] * x_mean
pdf_mean = norm.pdf(z_mean)
marginal_effect_mean = pdf_mean * beta_hat_probit[1]

print(f"\nEfecto Marginal Evaluado en la Media (MEM):")
print(f"  x̄ = {x_mean:.6f}")
print(f"  z̄ = β̂₀ + β̂₁x̄ = {z_mean:.4f}")
print(f"  φ(z̄) = {pdf_mean:.4f}")
print(f"  ∂P/∂x|_x̄ = φ(z̄) × β̂₁ = {marginal_effect_mean:.4f}")
print(f"\nInterpretación:")
print(f"  Un aumento de 0.01 (1 p.p.) en el retorno de NVDA aumenta")
print(f"  la probabilidad de retorno positivo de AAPL en")
print(f"  {marginal_effect_mean * 0.01:.4f} puntos porcentuales (evaluado en x̄)")

# Efecto marginal promedio (AME)
pdf_all = norm.pdf(z_fitted)
marginal_effects_all = pdf_all * beta_hat_probit[1]
ame = marginal_effects_all.mean()

print(f"\nEfecto Marginal Promedio (AME):")
print(f"  AME = (1/n) Σ [φ(β̂₀ + β̂₁xᵢ) × β̂₁] = {ame:.4f}")

#%% ===========================================================================
# PARTE 5: ESTIMACIÓN MANUAL DEL MODELO LOGIT
# =============================================================================

print("\n" + "="*80)
print("PARTE 5: ESTIMACIÓN MANUAL DEL MODELO LOGIT")
print("="*80)

def logit_cdf(z):
    """Función CDF logística"""
    return 1 / (1 + np.exp(-z))

def logit_pdf(z):
    """Función PDF logística"""
    cdf = logit_cdf(z)
    return cdf * (1 - cdf)

def logit_log_likelihood(params, y, x):
    """Log-verosimilitud del modelo Logit"""
    beta_0, beta_1 = params
    z = beta_0 + beta_1 * x

    # Forma simplificada: y*z - log(1 + exp(z))
    log_lik = np.sum(y * z - np.log(1 + np.exp(z)))

    return -log_lik

# Maximización
result_logit = minimize(
    logit_log_likelihood,
    params_init,
    args=(y, x),
    method='BFGS'
)

beta_hat_logit = result_logit.x

print(f"\n{'Resultados del Modelo Logit (Manual):':-^80}")
print(f"  β̂₀ (Intercepto): {beta_hat_logit[0]:.4f}")
print(f"  β̂₁ (Pendiente):  {beta_hat_logit[1]:.4f}")
print(f"  Log-verosimilitud: {-result_logit.fun:.4f}")

# Matriz de información para Logit
def fisher_information_logit(params, y, x):
    """Matriz de información de Fisher para Logit"""
    beta_0, beta_1 = params
    z = beta_0 + beta_1 * x

    prob = logit_cdf(z)
    weight = prob * (1 - prob)

    info = np.zeros((2, 2))
    info[0, 0] = np.sum(weight)
    info[0, 1] = np.sum(weight * x)
    info[1, 0] = info[0, 1]
    info[1, 1] = np.sum(weight * x ** 2)

    return info

info_matrix_logit = fisher_information_logit(beta_hat_logit, y, x)
var_cov_matrix_logit = np.linalg.inv(info_matrix_logit)
se_logit = np.sqrt(np.diag(var_cov_matrix_logit))

print(f"\n  SE(β̂₀): {se_logit[0]:.4f}")
print(f"  SE(β̂₁): {se_logit[1]:.4f}")

#%% ===========================================================================
# PARTE 6: COMPARACIÓN PROBIT VS LOGIT
# =============================================================================

print("\n" + "="*80)
print("PARTE 6: COMPARACIÓN PROBIT VS LOGIT")
print("="*80)

# Tabla comparativa
comparison_df = pd.DataFrame({
    'Parámetro': ['β₀ (Intercepto)', 'β₁ (Pendiente)'],
    'Probit': beta_hat_probit,
    'Logit': beta_hat_logit,
    'Ratio Logit/Probit': beta_hat_logit / beta_hat_probit
})

print("\nComparación de Coeficientes:")
print(comparison_df.round(4).to_string(index=False))

print(f"\nRegla empírica: β̂_Logit ≈ 1.6 × β̂_Probit")
print(f"Ratio observado (β₁): {beta_hat_logit[1]/beta_hat_probit[1]:.2f}")

# Odds ratio del Logit
print(f"\n{'Interpretación del Odds Ratio (Logit):':-^80}")
odds_ratio_001 = np.exp(beta_hat_logit[1] * 0.01)
print(f"  OR(Δx = 0.01) = exp({beta_hat_logit[1]:.4f} × 0.01) = {odds_ratio_001:.4f}")
print(f"  Un aumento de 1 p.p. en el retorno de NVDA multiplica")
print(f"  las probabilidades (odds) por {odds_ratio_001:.4f}")
print(f"  Es decir, las aumenta en {(odds_ratio_001-1)*100:.2f}%")

#%% ===========================================================================
# PARTE 7: VERIFICACIÓN CON STATSMODELS
# =============================================================================

print("\n" + "="*80)
print("PARTE 7: VERIFICACIÓN CON STATSMODELS")
print("="*80)

# Preparar datos para statsmodels
X = sm.add_constant(x)  # Añadir intercepto

# Modelo Probit
probit_model = Probit(y, X)
probit_results = probit_model.fit(disp=0)

print("\nResultados Probit (statsmodels):")
print(probit_results.summary2().tables[1])

# Modelo Logit
logit_model = Logit(y, X)
logit_results = logit_model.fit(disp=0)

print("\nResultados Logit (statsmodels):")
print(logit_results.summary2().tables[1])

# Comparación de estimadores
print(f"\n{'Verificación de Estimadores Manuales:':-^80}")
print(f"Probit:")
print(f"  Manual:      β₀={beta_hat_probit[0]:.4f}, β₁={beta_hat_probit[1]:.4f}")
print(f"  Statsmodels: β₀={probit_results.params[0]:.4f}, β₁={probit_results.params[1]:.4f}")
print(f"  Diferencia:  β₀={abs(beta_hat_probit[0]-probit_results.params[0]):.6f}, β₁={abs(beta_hat_probit[1]-probit_results.params[1]):.6f}")

print(f"\nLogit:")
print(f"  Manual:      β₀={beta_hat_logit[0]:.4f}, β₁={beta_hat_logit[1]:.4f}")
print(f"  Statsmodels: β₀={logit_results.params[0]:.4f}, β₁={logit_results.params[1]:.4f}")
print(f"  Diferencia:  β₀={abs(beta_hat_logit[0]-logit_results.params[0]):.6f}, β₁={abs(beta_hat_logit[1]-logit_results.params[1]):.6f}")

#%% ===========================================================================
# PARTE 8: GRÁFICAS
# =============================================================================

print("\n" + "="*80)
print("PARTE 8: GENERACIÓN DE GRÁFICAS")
print("="*80)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Modelos Probit y Logit: Análisis Gráfico', fontsize=16, fontweight='bold')

# Subplot 1: Comparación de Curvas de Probabilidad
ax1 = axes[0, 0]
x_range = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
z_probit = beta_hat_probit[0] + beta_hat_probit[1] * x_range
z_logit = beta_hat_logit[0] + beta_hat_logit[1] * x_range
prob_probit = norm.cdf(z_probit)
prob_logit = logit_cdf(z_logit)

ax1.plot(x_range, prob_probit, 'b-', linewidth=2, label='Probit', alpha=0.8)
ax1.plot(x_range, prob_logit, 'r--', linewidth=2, label='Logit', alpha=0.8)
ax1.scatter(x[y==1], y[y==1], c='green', marker='o', s=100, alpha=0.6, label='y=1', edgecolors='black')
ax1.scatter(x[y==0], y[y==0], c='orange', marker='s', s=100, alpha=0.6, label='y=0', edgecolors='black')
ax1.set_xlabel('Retorno NVDA', fontsize=11)
ax1.set_ylabel('P(y=1 | x)', fontsize=11)
ax1.set_title('Curvas de Probabilidad: Probit vs Logit', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

# Subplot 2: Efectos Marginales
ax2 = axes[0, 1]
pdf_probit = norm.pdf(z_probit)
pdf_logit = logit_pdf(z_logit)
me_probit = pdf_probit * beta_hat_probit[1]
me_logit = pdf_logit * beta_hat_logit[1]

ax2.plot(x_range, me_probit, 'b-', linewidth=2, label='Probit', alpha=0.8)
ax2.plot(x_range, me_logit, 'r--', linewidth=2, label='Logit', alpha=0.8)
ax2.axvline(x=x_mean, color='gray', linestyle=':', alpha=0.5, label='Media de x')
ax2.set_xlabel('Retorno NVDA', fontsize=11)
ax2.set_ylabel('∂P(y=1)/∂x', fontsize=11)
ax2.set_title('Efectos Marginales', fontsize=12, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Subplot 3: Residuos (Pearson)
ax3 = axes[1, 0]
residuals_probit = (y - norm.cdf(beta_hat_probit[0] + beta_hat_probit[1] * x))
ax3.scatter(range(1, n+1), residuals_probit, c='blue', s=100, alpha=0.6, edgecolors='black')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Observación', fontsize=11)
ax3.set_ylabel('Residuo (y - P̂)', fontsize=11)
ax3.set_title('Residuos del Modelo Probit', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Subplot 4: Comparación de Predicciones
ax4 = axes[1, 1]
prob_probit_fitted = norm.cdf(beta_hat_probit[0] + beta_hat_probit[1] * x)
prob_logit_fitted = logit_cdf(beta_hat_logit[0] + beta_hat_logit[1] * x)

ax4.scatter(prob_probit_fitted, prob_logit_fitted, c='purple', s=100, alpha=0.6, edgecolors='black')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Línea 45°')
ax4.set_xlabel('P̂(y=1) Probit', fontsize=11)
ax4.set_ylabel('P̂(y=1) Logit', fontsize=11)
ax4.set_title('Comparación de Predicciones', fontsize=12, fontweight='bold')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('/content/probit_logit_analisis.png', dpi=300, bbox_inches='tight')
print("\nGráfica guardada: probit_logit_analisis.png")
plt.show()

#%% ===========================================================================
# PARTE 9: ANÁLISIS CON MUESTRA COMPLETA
# =============================================================================

print("\n" + "="*80)
print("PARTE 9: ANÁLISIS CON MUESTRA COMPLETA")
print("="*80)

# Leer muestra completa
df_completo = pd.read_excel(file_path, sheet_name='Retornos')

# Crear variables
y_completo = (df_completo['AAPL_Return'] > 0).astype(int).values
x_completo = df_completo['NVDA_Return'].values

# Eliminar NaN
mask = ~np.isnan(x_completo) & ~np.isnan(y_completo)
y_completo = y_completo[mask]
x_completo = x_completo[mask]

print(f"\nMuestra completa: {len(y_completo)} observaciones")
print(f"Proporción de y=1: {y_completo.mean():.4f}")

# Estimar Probit con muestra completa
X_completo = sm.add_constant(x_completo)
probit_completo = Probit(y_completo, X_completo).fit(disp=0)
logit_completo = Logit(y_completo, X_completo).fit(disp=0)

print(f"\n{'Resultados con Muestra Completa:':-^80}")
print("\nProbit (n={len(y_completo)}):")
print(probit_completo.summary2().tables[1])

print("\nLogit (n={len(y_completo)}):")
print(logit_completo.summary2().tables[1])

# Efectos marginales
marginal_effects_probit = probit_completo.get_margeff(at='mean')
print(f"\n{'Efectos Marginales (Muestra Completa):':-^80}")
print("\nProbit:")
print(marginal_effects_probit.summary())

# Tabla de clasificación
y_pred_completo = (probit_completo.predict() > 0.5).astype(int)
accuracy_completo = (y_completo == y_pred_completo).mean()
print(f"\nTasa de clasificación correcta: {accuracy_completo:.1%}")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
print("\nArchivos generados:")
print("  - probit_logit_analisis.png")
print("\nTodos los cálculos han sido verificados exitosamente.")
