#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ECONOMETRÍA II - Ejemplo 01: Cálculo de Retornos Financieros
================================================================================

Archivo: ejemplo01_calculo_retornos.py
Capítulo: 1 - Introducción a la Econometría Financiera
Sección: 1.4 - Ejemplo Numérico: Cálculo de Retornos

Autor: Prof. Jeel Elvis Cueva Laguna
Institución: Universidad Nacional Hermilio Valdizán - Huánuco
Curso: Econometría II
Email: ecueva@unheval.edu.pe

Descripción:
    Este script calcula retornos simples y logarítmicos de precios de acciones,
    verifica la aproximación de Taylor, y demuestra la propiedad de aditividad
    temporal de los retornos logarítmicos.

Datos:
    Precios de cierre de Apple Inc. (AAPL) durante 5 días consecutivos en
    octubre de 2025.

Requisitos:
    - Python 3.7+
    - numpy
    - pandas

Instalación de dependencias:
    pip install numpy pandas

Uso:
    python ejemplo01_calculo_retornos.py

Fecha de creación: Enero 2026
Última modificación: Enero 2026
Versión: 1.0

Licencia: MIT License
================================================================================
"""

import numpy as np
import pandas as pd
import sys

# Configuración de visualización de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 8)

# ============================================================================
# DATOS DEL EJEMPLO
# ============================================================================
print("\n" + "="*70)
print("ECONOMETRÍA II - EJEMPLO 01: CÁLCULO DE RETORNOS")
print("="*70)
print("\nProf. Jeel Elvis Cueva Laguna")
print("Universidad Nacional Hermilio Valdizán - Huánuco\n")

# Datos de precios de Apple (AAPL) - Últimos 5 días de octubre 2025
fechas = pd.date_range(start='2025-10-27', periods=5, freq='D')
precios = np.array([1058.84, 1023.02, 1076.40, 1092.64, 1069.53])

# Crear DataFrame para organizar los datos
df = pd.DataFrame({
    'Precio': precios
}, index=fechas)

print("="*70)
print("DATOS ORIGINALES")
print("="*70)
print("\nPrecios de cierre de Apple Inc. (AAPL)")
print("Periodo: 27-31 de octubre, 2025\n")
print(df)
print(f"\nNúmero de observaciones: {len(df)}")
print(f"Precio inicial: ${df['Precio'].iloc[0]:.2f}")
print(f"Precio final:   ${df['Precio'].iloc[-1]:.2f}")
print(f"Cambio total:   ${df['Precio'].iloc[-1] - df['Precio'].iloc[0]:.2f}")
print()

# ============================================================================
# CALCULAR RETORNOS SIMPLES
# ============================================================================
print("="*70)
print("PARTE 1: CÁLCULO DE RETORNOS SIMPLES")
print("="*70)
print("\nFórmula: R_t = (P_t - P_{t-1}) / P_{t-1}\n")

# Método 1: Usando la fórmula manualmente
df['R_simple_manual'] = (df['Precio'] - df['Precio'].shift(1)) / df['Precio'].shift(1)

# Método 2: Usando función de pandas (más eficiente)
df['R_simple_pandas'] = df['Precio'].pct_change()

# Verificar que ambos métodos dan el mismo resultado
print("Comparación de métodos de cálculo:")
print(df[['Precio', 'R_simple_manual', 'R_simple_pandas']])
print(f"\n¿Los métodos coinciden? {np.allclose(df['R_simple_manual'].dropna(), df['R_simple_pandas'].dropna())}")

if np.allclose(df['R_simple_manual'].dropna(), df['R_simple_pandas'].dropna()):
    print("✓ Verificado: Ambos métodos producen resultados idénticos")
else:
    print("✗ Error: Los métodos producen resultados diferentes")
    sys.exit(1)

print()

# ============================================================================
# CALCULAR RETORNOS LOGARÍTMICOS
# ============================================================================
print("="*70)
print("PARTE 2: CÁLCULO DE RETORNOS LOGARÍTMICOS")
print("="*70)
print("\nFórmula: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})\n")

# Método 1: Usando logaritmo del ratio de precios
df['r_log_manual'] = np.log(df['Precio'] / df['Precio'].shift(1))

# Método 2: Diferencia de logaritmos
df['r_log_diff'] = np.log(df['Precio']) - np.log(df['Precio'].shift(1))

# Método 3: Usando la relación con retorno simple: r = ln(1 + R)
df['r_log_from_simple'] = np.log(1 + df['R_simple_pandas'])

print("Comparación de tres métodos de cálculo:")
print(df[['Precio', 'r_log_manual', 'r_log_diff', 'r_log_from_simple']])

# Verificar que los tres métodos dan el mismo resultado
metodo1_vs_metodo2 = np.allclose(df['r_log_manual'].dropna(), df['r_log_diff'].dropna())
metodo1_vs_metodo3 = np.allclose(df['r_log_manual'].dropna(), df['r_log_from_simple'].dropna())

print(f"\n¿Los tres métodos coinciden? {metodo1_vs_metodo2 and metodo1_vs_metodo3}")

if metodo1_vs_metodo2 and metodo1_vs_metodo3:
    print("✓ Verificado: Los tres métodos producen resultados idénticos")
else:
    print("✗ Error: Los métodos producen resultados diferentes")
    sys.exit(1)

print()

# ============================================================================
# CALCULAR DIFERENCIAS Y APROXIMACIONES
# ============================================================================
print("="*70)
print("PARTE 3: COMPARACIÓN Y APROXIMACIÓN DE TAYLOR")
print("="*70)
print("\nAproximación de Taylor: r ≈ R - R²/2 + R³/3 - ...\n")

# Diferencia entre retorno simple y logarítmico
df['Diferencia'] = np.abs(df['R_simple_pandas'] - df['r_log_manual'])

# Aproximación de Taylor de segundo orden: r ≈ R - R²/2
df['r_aprox_taylor'] = df['R_simple_pandas'] - (df['R_simple_pandas']**2)/2

# Error de la aproximación de Taylor
df['error_taylor'] = np.abs(df['r_log_manual'] - df['r_aprox_taylor'])

print("Comparación de retornos y aproximaciones:")
resultados = df[['R_simple_pandas', 'r_log_manual', 'Diferencia', 
                 'r_aprox_taylor', 'error_taylor']]
print(resultados)
print()

# ============================================================================
# ESTADÍSTICAS DE LAS DIFERENCIAS
# ============================================================================
print("="*70)
print("ESTADÍSTICAS DE LAS DIFERENCIAS Y ERRORES")
print("="*70)
print("\nDiferencia |R - r| (aproximación de primer orden):")
print(f"  Promedio:  {df['Diferencia'].mean():.8f}")
print(f"  Máximo:    {df['Diferencia'].max():.8f}")
print(f"  Mínimo:    {df['Diferencia'].min():.8f}")
print()
print("Error de aproximación de Taylor de segundo orden:")
print(f"  Promedio:  {df['error_taylor'].mean():.8f}")
print(f"  Máximo:    {df['error_taylor'].max():.8f}")
print(f"  Mínimo:    {df['error_taylor'].min():.8f}")
print()

# Calcular mejora porcentual
mejora = (1 - df['error_taylor'].mean() / df['Diferencia'].mean()) * 100
print(f"Reducción de error con aproximación cuadrática: {mejora:.1f}%")
print()

# ============================================================================
# VERIFICAR ADITIVIDAD DE RETORNOS LOGARÍTMICOS
# ============================================================================
print("="*70)
print("PARTE 4: VERIFICACIÓN DE ADITIVIDAD TEMPORAL")
print("="*70)
print("\nPropiedad teórica: r_{t,t+k} = Σ r_{t+i}\n")

# Retorno acumulado del día 2 al 5 (sumando retornos logarítmicos)
r_acum_suma = df['r_log_manual'].iloc[1:].sum()
print(f"Método 1 - Suma de retornos log (día 2-5): {r_acum_suma:.8f}")

# Retorno acumulado calculado directamente
r_acum_directo = np.log(df['Precio'].iloc[-1] / df['Precio'].iloc[0])
print(f"Método 2 - Retorno log directo (día 1-5):  {r_acum_directo:.8f}")

# Diferencia (debe ser cero salvo redondeo)
diferencia_log = np.abs(r_acum_suma - r_acum_directo)
print(f"Diferencia:                                 {diferencia_log:.10f}")
print(f"\n¿Se verifica la aditividad? {np.isclose(r_acum_suma, r_acum_directo)}")

if np.isclose(r_acum_suma, r_acum_directo):
    print(f"✓ Verificado: Retornos logarítmicos son aditivos")
    print(f"  (diferencia {diferencia_log:.2e} se debe a redondeo numérico)")
else:
    print("✗ Error: Los retornos logarítmicos no son aditivos")

print()
print("-" * 70)
print("Verificación con retornos simples (NO aditivos):")
print("-" * 70)

# Para retornos simples (NO aditivos)
R_suma = df['R_simple_pandas'].iloc[1:].sum()
R_directo = (df['Precio'].iloc[-1] - df['Precio'].iloc[0]) / df['Precio'].iloc[0]
diferencia_simple = np.abs(R_suma - R_directo)

print(f"\nMétodo 1 - Suma de retornos simples:        {R_suma:.8f}")
print(f"Método 2 - Retorno simple directo:          {R_directo:.8f}")
print(f"Diferencia:                                  {diferencia_simple:.8f}")
print(f"\n¿Son aditivos? {np.isclose(R_suma, R_directo)}")

if not np.isclose(R_suma, R_directo):
    print("✓ Confirmado: Retornos simples NO son aditivos")
    print(f"  (diferencia de {diferencia_simple*100:.3f}% es significativa)")
else:
    print("✗ Error inesperado: Los retornos parecen aditivos")

print()
print("Fórmula correcta para retornos simples:")
# Forma correcta para retornos simples: producto de (1+R)
R_producto = np.prod(1 + df['R_simple_pandas'].iloc[1:]) - 1
print(f"Producto Π(1+R_t) - 1:                      {R_producto:.8f}")
print(f"Retorno directo:                             {R_directo:.8f}")
print(f"¿Coinciden? {np.isclose(R_producto, R_directo)}")

if np.isclose(R_producto, R_directo):
    print("✓ Verificado: La fórmula correcta es R_{acum} = Π(1+R_t) - 1")
else:
    print("✗ Error: La fórmula del producto no funciona")

print()

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("="*70)
print("RESUMEN COMPARATIVO FINAL")
print("="*70)
print()

resumen = pd.DataFrame({
    'Fecha': ['Oct 28', 'Oct 29', 'Oct 30', 'Oct 31'],
    'Precio': df['Precio'].iloc[1:].values,
    'R_simple': df['R_simple_pandas'].iloc[1:].values,
    'r_log': df['r_log_manual'].iloc[1:].values,
    'Diferencia': df['Diferencia'].iloc[1:].values
})

print(resumen.to_string(index=False))
print()

# ============================================================================
# CONCLUSIONES
# ============================================================================
print("="*70)
print("CONCLUSIONES")
print("="*70)
print()
print("✓ Todos los cálculos manuales han sido verificados con éxito")
print("✓ Múltiples métodos de cálculo producen resultados idénticos")
print("✓ La aproximación r ≈ R es excelente para retornos pequeños")
print("✓ La aproximación de Taylor de 2° orden reduce el error en {:.1f}%".format(mejora))
print("✓ Los retornos logarítmicos son temporalmente aditivos")
print("✓ Los retornos simples NO son aditivos (requieren producto)")
print()
print("="*70)
print("Fin del análisis")
print("="*70)
print()
