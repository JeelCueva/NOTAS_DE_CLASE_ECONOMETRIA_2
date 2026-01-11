# 🚀 GUÍA DE USO INMEDIATO - NOTAS_DE_CLASE_ECONOMETRIA_2

## ✅ TODO LISTO PARA USAR

Tu repositorio está **100% completo y estructurado**. Esta guía te muestra cómo usarlo INMEDIATAMENTE.

---

## 📁 ¿QUÉ TIENES?

```
NOTAS_DE_CLASE_ECONOMETRIA_2/
├── README.md ✅                      # Documentación principal
├── LICENSE ✅                        # Licencia MIT
├── .gitignore ✅                     # Archivos ignorados
├── requirements.txt ✅               # Dependencias Python
├── inicializar.sh ✅                 # Script de configuración rápida
│
├── capitulo01/ ✅
│   ├── README.md
│   ├── ejemplo01_calculo_retornos.py ✅ (LISTO PARA EJECUTAR)
│   └── ejercicios/
│
├── capitulo02/ ✅ (estructura lista)
├── capitulo03/ ✅ (estructura lista)
├── capitulo04/ ✅ (estructura lista)
├── capitulo05/ ✅ (estructura lista)
├── capitulo06/ ✅ (estructura lista)
│
├── datos/ ✅
│   ├── README.md
│   ├── datos_activos_financieros.xlsx ✅
│   └── datos_activos_financieros.csv ✅
│
├── documentos/ ✅
│   └── preamble_actualizado.tex ✅
│
├── imagenes/ ✅ (carpeta lista para gráficas)
└── utils/ ✅ (carpeta lista para utilidades)
```

---

## 🎯 OPCIÓN 1: USO LOCAL (Sin GitHub)

### Paso 1: Instalar Python (si no lo tienes)

**Windows:**
```bash
# Descargar desde: https://www.python.org/downloads/
# Durante instalación: marcar "Add Python to PATH"
```

**Mac:**
```bash
brew install python3
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip
```

### Paso 2: Instalar Dependencias

```bash
# Navegar a la carpeta
cd NOTAS_DE_CLASE_ECONOMETRIA_2

# Instalar librerías
pip install -r requirements.txt
```

### Paso 3: Ejecutar un Ejemplo

```bash
# Ir al Capítulo 1
cd capitulo01

# Ejecutar el script
python ejemplo01_calculo_retornos.py
```

**¡Deberías ver los resultados inmediatamente!** ✅

---

## 🌐 OPCIÓN 2: SUBIR A GITHUB (Recomendado)

### Paso 1: Crear Repositorio en GitHub

1. Ve a: https://github.com/new
2. **Nombre:** `NOTAS_DE_CLASE_ECONOMETRIA_2`
3. **Descripción:** "Material didáctico para Econometría II - UNHEVAL"
4. **Visibilidad:** Público (para que los estudiantes accedan)
5. **NO inicializar** con README (ya lo tienes)
6. Click en "Create repository"

### Paso 2: Usar el Script de Inicialización Rápida

```bash
# Navegar a la carpeta
cd NOTAS_DE_CLASE_ECONOMETRIA_2

# Dar permisos de ejecución al script
chmod +x inicializar.sh

# Ejecutar el script
./inicializar.sh
```

El script automáticamente:
- ✅ Verifica Git
- ✅ Inicializa el repositorio
- ✅ Configura el remoto
- ✅ Verifica Python
- ✅ Ofrece instalar dependencias
- ✅ Crea el primer commit

### Paso 3: Subir a GitHub

```bash
# Cambiar el nombre de la rama a 'main' (si es necesario)
git branch -M main

# Subir todo
git push -u origin main
```

### Paso 4: Verificar en GitHub

Visita: `https://github.com/TU_USUARIO/NOTAS_DE_CLASE_ECONOMETRIA_2`

¡Deberías ver todo el repositorio! 🎉

---

## 📝 INTEGRACIÓN CON TU DOCUMENTO LaTeX

### Paso 1: Actualizar Preámbulo

1. Abre tu archivo: `notas_econometria_adaptado.tex`

2. Busca la sección de comandos GitHub (línea ~135):

```latex
% ============================================================================
% COMANDOS GITHUB
% ============================================================================
```

3. **REEMPLAZA** los 3 comandos existentes con estos:

```latex
\newcommand{\scriptPython}[2]{%
    \href{https://github.com/JeelCueva/NOTAS_DE_CLASE_ECONOMETRIA_2/blob/main/#1}{%
        \textcolor{pythoncolor}{\nolinkurl{#2}}%
    }%
}

\newcommand{\scriptR}[2]{%
    \href{https://github.com/JeelCueva/NOTAS_DE_CLASE_ECONOMETRIA_2/blob/main/#1}{%
        \textcolor{rcolor}{\nolinkurl{#2}}%
    }%
}

\newcommand{\repoGitHub}{%
    \url{https://github.com/JeelCueva/NOTAS_DE_CLASE_ECONOMETRIA_2}%
}
```

4. **Guarda** el archivo

### Paso 2: Usar en tu Documento

Ahora puedes usar estos comandos:

```latex
% Enlace a script de Python
El código está en: \scriptPython{capitulo01/ejemplo01_calculo_retornos.py}{ejemplo01\_calculo\_retornos.py}

% Enlace al repositorio
Repositorio completo: \repoGitHub
```

### Paso 3: Compilar PDF

```bash
pdflatex notas_econometria_adaptado.tex
biber notas_econometria_adaptado
pdflatex notas_econometria_adaptado.tex
pdflatex notas_econometria_adaptado.tex
```

**¡Los enlaces estarán clickables y apuntarán a tu GitHub!** ✅

---

## 🧪 PROBAR QUE TODO FUNCIONA

### Test 1: Ejecutar Script Local

```bash
cd capitulo01
python ejemplo01_calculo_retornos.py
```

**Resultado esperado:** Ver estadísticas y verificaciones

### Test 2: Verificar Datos

```bash
cd datos
python -c "import pandas as pd; df = pd.read_excel('datos_activos_financieros.xlsx', sheet_name='Precios'); print(df.head())"
```

**Resultado esperado:** Ver las primeras 5 filas de precios

### Test 3: Verificar GitHub (si subiste)

Visita: `https://github.com/TU_USUARIO/NOTAS_DE_CLASE_ECONOMETRIA_2/blob/main/capitulo01/ejemplo01_calculo_retornos.py`

**Resultado esperado:** Ver el código en GitHub

### Test 4: Verificar Enlaces PDF

1. Compila tu documento LaTeX
2. Abre el PDF
3. Click en un enlace de GitHub
4. Debería abrirse la página correcta

---

## 💡 AGREGAR MÁS CONTENIDO

### Agregar un Nuevo Script

```bash
# Crear el script
cd capitulo02
nano ejemplo01_regresion_simple.py

# Agregar a Git
git add ejemplo01_regresion_simple.py
git commit -m "Agregar ejemplo de regresión simple"
git push
```

### Agregar Más Datos

```bash
# Copiar archivo a la carpeta datos
cp mi_nueva_data.xlsx datos/

# Agregar a Git
git add datos/mi_nueva_data.xlsx
git commit -m "Agregar nueva base de datos"
git push
```

### Actualizar README

```bash
# Editar README.md
nano README.md

# Guardar cambios
git add README.md
git commit -m "Actualizar documentación"
git push
```

---

## 🔧 COMANDOS GIT ÚTILES

```bash
# Ver estado
git status

# Ver historial
git log --oneline

# Ver archivos modificados
git diff

# Deshacer cambios locales
git checkout -- archivo.py

# Actualizar desde GitHub
git pull

# Crear una rama nueva
git checkout -b nueva-rama

# Cambiar de rama
git checkout main
```

---

## 📊 ESTRUCTURA DE UN SCRIPT TÍPICO

Todos tus scripts deben seguir esta estructura:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ECONOMETRÍA II - Capítulo X: Título
================================================================================

Archivo: ejemploXX_nombre.py
Descripción: Breve descripción de qué hace el script

Autor: Prof. Jeel Elvis Cueva Laguna
Email: ecueva@unheval.edu.pe
================================================================================
"""

import numpy as np
import pandas as pd

# ============================================================================
# SECCIÓN 1: CARGA DE DATOS
# ============================================================================

# Código aquí

# ============================================================================
# SECCIÓN 2: ANÁLISIS
# ============================================================================

# Código aquí

# ============================================================================
# SECCIÓN 3: RESULTADOS
# ============================================================================

# Código aquí

if __name__ == "__main__":
    print("Script ejecutado correctamente")
```

---

## ⚠️ PROBLEMAS COMUNES Y SOLUCIONES

### Problema 1: "ModuleNotFoundError"

**Causa:** Falta una librería

**Solución:**
```bash
pip install nombre_libreria
# o
pip install -r requirements.txt
```

### Problema 2: "FileNotFoundError: datos_activos_financieros.xlsx"

**Causa:** El script no encuentra los datos

**Solución:**
```bash
# Verificar que estás en la carpeta correcta
cd capitulo01

# El archivo debe estar en ../datos/
ls ../datos/
```

### Problema 3: Git push falla

**Causa:** No tienes permisos o no configuraste GitHub

**Solución:**
```bash
# Configurar credenciales
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"

# Usar token de acceso personal en lugar de contraseña
# Ver: https://github.com/settings/tokens
```

### Problema 4: Enlaces de LaTeX no funcionan

**Causa:** El repositorio no es público o el nombre no coincide

**Solución:**
1. Hacer el repositorio público en GitHub
2. Verificar que el nombre sea exactamente: `NOTAS_DE_CLASE_ECONOMETRIA_2`
3. Actualizar el preámbulo LaTeX con tu usuario correcto

---

## 📚 PRÓXIMOS PASOS

1. ✅ **Familiarízate** con la estructura
2. ✅ **Ejecuta** el ejemplo del Capítulo 1
3. ✅ **Sube** a GitHub (opcional pero recomendado)
4. ✅ **Integra** con tu documento LaTeX
5. ✅ **Agrega** más contenido según avances en el curso
6. ✅ **Comparte** con tus estudiantes

---

## 📧 SOPORTE

Si tienes problemas:

1. **Revisa** esta guía completa
2. **Consulta** el README.md principal
3. **Busca** en Google el error específico
4. **Contacta** a ecueva@unheval.edu.pe

---

## 🎉 ¡FELICIDADES!

Tu repositorio está **100% listo para usar**. 

No necesitas configurar nada más. Solo:

1. Ejecuta los scripts
2. Modifica según necesites
3. Agrega más contenido
4. Comparte con estudiantes

**¡A enseñar Econometría! 📊📈🎓**

---

**Última actualización:** Enero 2026  
**Versión:** 1.0
