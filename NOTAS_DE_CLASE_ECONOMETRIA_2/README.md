# 📚 NOTAS DE CLASE: ECONOMETRÍA II

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![R](https://img.shields.io/badge/R-4.0%2B-blue)
![LaTeX](https://img.shields.io/badge/LaTeX-Document-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Material Didáctico Completo para el Curso de Econometría II**

Universidad Nacional Hermilio Valdizán - Huánuco  
Facultad de Economía

</div>

---

## 👨‍🏫 Información del Curso

- **Curso:** Econometría II
- **Código:** 3204
- **Docente:** Prof. Jeel Elvis Cueva Laguna
- **Email:** ecueva@unheval.edu.pe
- **Ciclo:** Verano 2026
- **Periodo:** Enero - Febrero 2026
- **Modalidad:** Presencial
- **Horario:** Lunes y Martes, 8:00 AM - 12:00 PM

---

## 📖 Descripción

Este repositorio contiene **todo el material computacional** del curso de Econometría II:

✅ **Scripts Python** con ejemplos completos  
✅ **Base de datos** de activos financieros  
✅ **Notebooks interactivos** (próximamente)  
✅ **Código R** (opcional)  
✅ **Documentación completa**  
✅ **Ejercicios resueltos**

---

## 📂 Estructura del Repositorio

```
NOTAS_DE_CLASE_ECONOMETRIA_2/
│
├── README.md                          # Este archivo
├── LICENSE                            # Licencia MIT
├── .gitignore                         # Archivos ignorados por Git
├── requirements.txt                   # Dependencias de Python
│
├── capitulo01/                        # Introducción a Econometría Financiera
│   ├── README.md
│   ├── ejemplo01_calculo_retornos.py
│   ├── ejemplo02_estadisticas_descriptivas.py
│   └── ejercicios/
│
├── capitulo02/                        # Regresión Lineal Múltiple
│   ├── README.md
│   ├── ejemplo01_regresion_simple.py
│   ├── ejemplo02_regresion_multiple.py
│   ├── ejemplo03_diagnosticos.py
│   └── ejercicios/
│
├── capitulo03/                        # Variable Dependiente Limitada
│   ├── README.md
│   ├── ejemplo01_probit_logit.py
│   ├── ejemplo02_comparacion_modelos.py
│   └── ejercicios/
│
├── capitulo04/                        # Series de Tiempo
│   ├── README.md
│   ├── ejemplo01_ar_ma_arma.py
│   ├── ejemplo02_arima.py
│   ├── ejemplo03_raiz_unitaria.py
│   └── ejercicios/
│
├── capitulo05/                        # Modelos Multivariados
│   ├── README.md
│   ├── ejemplo01_var.py
│   ├── ejemplo02_cointegracion.py
│   ├── ejemplo03_vec.py
│   └── ejercicios/
│
├── capitulo06/                        # Volatilidad y Panel
│   ├── README.md
│   ├── ejemplo01_arch_garch.py
│   ├── ejemplo02_panel_efectos_fijos.py
│   ├── ejemplo03_panel_efectos_aleatorios.py
│   └── ejercicios/
│
├── datos/                             # Bases de datos
│   ├── README.md
│   ├── datos_activos_financieros.xlsx
│   ├── datos_activos_financieros.csv
│   └── descripcion_variables.md
│
├── documentos/                        # Documentación adicional
│   ├── README.md
│   ├── guia_instalacion.md
│   ├── guia_python.md
│   ├── guia_R.md
│   └── referencias_bibliograficas.md
│
├── imagenes/                          # Gráficas y figuras
│   ├── README.md
│   └── (gráficas generadas por los scripts)
│
└── utils/                             # Utilidades y funciones auxiliares
    ├── README.md
    ├── funciones_comunes.py
    └── config.py
```

---

## 🚀 Inicio Rápido

### 1. Clonar el Repositorio

```bash
git clone https://github.com/JeelCueva/NOTAS_DE_CLASE_ECONOMETRIA_2.git
cd NOTAS_DE_CLASE_ECONOMETRIA_2
```

### 2. Instalar Dependencias

**Python:**
```bash
pip install -r requirements.txt
```

**R (opcional):**
```R
install.packages(c("readxl", "ggplot2", "dplyr", "tidyr", 
                   "forecast", "tseries", "urca", "vars"))
```

### 3. Ejecutar un Ejemplo

```bash
cd capitulo01
python ejemplo01_calculo_retornos.py
```

---

## 📊 Datos Incluidos

### Base de Datos de Activos Financieros

**Archivo:** `datos/datos_activos_financieros.xlsx`

**Contenido:**
- 8 activos financieros: AAPL, GOOGL, TSLA, MSFT, AMZN, META, NVDA, JPM
- 1,260 observaciones diarias (≈5 años)
- Periodo: 2021-01-04 a 2025-10-31
- Variables: Precios, Retornos, Volatilidad

**Hojas del archivo:**
1. **Precios** - Precios diarios de cierre ajustados
2. **Retornos** - Retornos logarítmicos calculados
3. **Volatilidad** - Volatilidad condicional GARCH(1,1)
4. **Estadisticas** - Resumen descriptivo
5. **Muestra_20dias** - Para ejemplos en clase
6. **Muestra_Manual_10obs** - Para cálculos manuales

---

## 💻 Requisitos del Sistema

### Software Necesario

| Software | Versión | Propósito |
|----------|---------|-----------|
| Python | 3.8+ | Scripts principales |
| R | 4.0+ | Scripts opcionales |
| Git | 2.0+ | Control de versiones |

### Librerías Python

```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.13.0
arch>=5.0.0
openpyxl>=3.0.0
```

---

## 📚 Contenido por Capítulo

### 📘 Capítulo 1: Introducción a la Econometría Financiera

- Naturaleza de los datos financieros
- Precios y retornos
- Propiedades de retornos logarítmicos
- Estadísticas descriptivas

**Scripts disponibles:**
- `ejemplo01_calculo_retornos.py` - Cálculo de retornos simples y logarítmicos
- `ejemplo02_estadisticas_descriptivas.py` - Análisis descriptivo completo

### 📗 Capítulo 2: Regresión Lineal Múltiple

- Modelo clásico de regresión
- Estimación por MCO
- Propiedades del estimador
- Diagnósticos y validación

**Scripts disponibles:**
- `ejemplo01_regresion_simple.py` - Regresión con 2 variables
- `ejemplo02_regresion_multiple.py` - Regresión con k variables
- `ejemplo03_diagnosticos.py` - Análisis de residuos

### 📙 Capítulo 3: Modelos para Variable Dependiente Limitada

- Modelo de Probabilidad Lineal
- Modelos Probit y Logit
- Comparación de modelos
- Efectos marginales

**Scripts disponibles:**
- `ejemplo01_probit_logit.py` - Estimación de modelos
- `ejemplo02_comparacion_modelos.py` - Comparación y validación

### 📕 Capítulo 4: Series de Tiempo Univariadas

- Modelos AR, MA, ARMA
- Modelos ARIMA
- Pruebas de raíz unitaria
- Pronóstico

**Scripts disponibles:**
- `ejemplo01_ar_ma_arma.py` - Modelos básicos
- `ejemplo02_arima.py` - Modelos ARIMA
- `ejemplo03_raiz_unitaria.py` - Tests ADF, PP, KPSS

### 📔 Capítulo 5: Modelos Multivariados

- Vectores Autorregresivos (VAR)
- Cointegración (Engle-Granger, Johansen)
- Modelos VEC

**Scripts disponibles:**
- `ejemplo01_var.py` - Estimación VAR
- `ejemplo02_cointegracion.py` - Tests de cointegración
- `ejemplo03_vec.py` - Modelos VEC

### 📓 Capítulo 6: Volatilidad y Datos de Panel

- Modelos ARCH/GARCH
- Variantes: EGARCH, TGARCH
- Datos de panel: efectos fijos y aleatorios
- Test de Hausman

**Scripts disponibles:**
- `ejemplo01_arch_garch.py` - Modelos de volatilidad
- `ejemplo02_panel_efectos_fijos.py` - Panel con EF
- `ejemplo03_panel_efectos_aleatorios.py` - Panel con EA

---

## 🎯 Cómo Usar Este Repositorio

### Para Estudiantes

1. **Clonar** el repositorio en tu computadora
2. **Instalar** las dependencias necesarias
3. **Seguir** los ejemplos en orden
4. **Ejecutar** los scripts para verificar resultados
5. **Modificar** el código para experimentar
6. **Resolver** los ejercicios propuestos

### Para Docentes

1. **Fork** este repositorio
2. **Personalizar** con tus propios ejemplos
3. **Agregar** material adicional
4. **Compartir** con tus estudiantes
5. **Actualizar** según necesidades del curso

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Si encuentras errores o tienes sugerencias:

1. Abre un **Issue** describiendo el problema o sugerencia
2. Haz un **Fork** del repositorio
3. Crea una **rama** para tu cambio
4. Haz un **Pull Request**

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2026 Jeel Elvis Cueva Laguna

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 📧 Contacto

**Prof. Jeel Elvis Cueva Laguna**  
📧 Email: ecueva@unheval.edu.pe  
🏛️ Universidad Nacional Hermilio Valdizán  
📍 Huánuco, Perú

---

## 🔗 Enlaces Útiles

- [Documento LaTeX completo](documentos/) - Notas de clase en PDF
- [Guía de instalación](documentos/guia_instalacion.md) - Configuración paso a paso
- [Referencias bibliográficas](documentos/referencias_bibliograficas.md) - Material complementario

---

## 📊 Estado del Proyecto

| Capítulo | Scripts | Ejercicios | Documentación | Estado |
|----------|---------|------------|---------------|--------|
| Cap. 1 | ✅ | ✅ | ✅ | Completo |
| Cap. 2 | ✅ | 🔄 | ✅ | En desarrollo |
| Cap. 3 | 🔄 | ⏳ | 🔄 | En desarrollo |
| Cap. 4 | ⏳ | ⏳ | ⏳ | Planeado |
| Cap. 5 | ⏳ | ⏳ | ⏳ | Planeado |
| Cap. 6 | ⏳ | ⏳ | ⏳ | Planeado |

✅ Completo | 🔄 En desarrollo | ⏳ Planeado

---

## 📅 Última Actualización

**Fecha:** Enero 2026  
**Versión:** 1.0.0

---

<div align="center">

**⭐ Si este material te es útil, por favor dale una estrella ⭐**

**Desarrollado con ❤️ para estudiantes de Econometría**

[Reportar un problema](../../issues) · [Solicitar una característica](../../issues)

</div>
