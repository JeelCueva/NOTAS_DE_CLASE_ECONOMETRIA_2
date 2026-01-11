# 📊 Carpeta de Datos

## Archivos Disponibles

### `datos_activos_financieros.xlsx`
**Formato:** Excel con múltiples hojas  
**Tamaño:** ~427 KB  
**Observaciones:** 1,260 por activo

**Hojas del archivo:**
1. **Precios** - Precios de cierre ajustados
2. **Retornos** - Retornos logarítmicos
3. **Volatilidad** - Volatilidad condicional GARCH
4. **Estadisticas** - Resumen descriptivo
5. **Muestra_20dias** - Para ejemplos en clase
6. **Muestra_Manual_10obs** - Para cálculos manuales

### `datos_activos_financieros.csv`
**Formato:** CSV  
**Tamaño:** ~605 KB  
**Contenido:** Todos los datos en formato plano

## Activos Incluidos

| Ticker | Empresa | Sector | Características |
|--------|---------|--------|-----------------|
| AAPL | Apple Inc. | Tecnología | Alta cap. |
| GOOGL | Alphabet Inc. | Tecnología | Estable |
| TSLA | Tesla Inc. | Automotriz | Alta vol. |
| MSFT | Microsoft | Tecnología | Blue chip |
| AMZN | Amazon | Comercio | E-commerce |
| META | Meta | Tecnología | Redes sociales |
| NVDA | NVIDIA | Semiconductores | Alto crecimiento |
| JPM | JPMorgan | Financiero | Banca |

## Características de los Datos

- **Periodo:** 2021-01-04 a 2025-10-31
- **Frecuencia:** Diaria (días hábiles)
- **Total observaciones:** 1,260 por activo
- **Generación:** Simulación GARCH(1,1)

## Cómo Usar los Datos

### Python:
```python
import pandas as pd

# Leer Excel
df = pd.read_excel('datos_activos_financieros.xlsx', 
                   sheet_name='Precios', 
                   index_col=0, 
                   parse_dates=True)

# Leer CSV
df = pd.read_csv('datos_activos_financieros.csv', 
                 index_col=0, 
                 parse_dates=True)
```

### R:
```r
library(readxl)

# Leer Excel
df <- read_excel("datos_activos_financieros.xlsx", 
                 sheet = "Precios")

# Leer CSV
df <- read.csv("datos_activos_financieros.csv")
```

## Descripción de Variables

Ver archivo `descripcion_variables.md` para detalles completos de cada variable.
