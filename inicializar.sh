#!/bin/bash
# ============================================================================
# SCRIPT DE INICIALIZACIÓN RÁPIDA
# NOTAS_DE_CLASE_ECONOMETRIA_2
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║     INICIALIZACIÓN DE REPOSITORIO: NOTAS_DE_CLASE_ECONOMETRIA_2     ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# PASO 1: VERIFICAR GIT
# ============================================================================
echo "📋 Paso 1: Verificando Git..."
if command -v git &> /dev/null; then
    echo "✓ Git está instalado: $(git --version)"
else
    echo "✗ Git NO está instalado. Por favor instala Git primero."
    exit 1
fi
echo ""

# ============================================================================
# PASO 2: INICIALIZAR REPOSITORIO
# ============================================================================
echo "📋 Paso 2: Inicializando repositorio Git..."
git init
echo "✓ Repositorio Git inicializado"
echo ""

# ============================================================================
# PASO 3: CONFIGURAR REMOTO
# ============================================================================
echo "📋 Paso 3: Configurando repositorio remoto..."
read -p "Ingresa tu usuario de GitHub: " github_user

if [ -z "$github_user" ]; then
    github_user="JeelCueva"
    echo "Usando usuario por defecto: $github_user"
fi

git remote add origin https://github.com/${github_user}/NOTAS_DE_CLASE_ECONOMETRIA_2.git
echo "✓ Remoto configurado: https://github.com/${github_user}/NOTAS_DE_CLASE_ECONOMETRIA_2"
echo ""

# ============================================================================
# PASO 4: VERIFICAR PYTHON
# ============================================================================
echo "📋 Paso 4: Verificando Python..."
if command -v python3 &> /dev/null; then
    echo "✓ Python está instalado: $(python3 --version)"
    
    echo ""
    read -p "¿Deseas instalar las dependencias de Python ahora? (s/n): " install_deps
    
    if [[ $install_deps == "s" || $install_deps == "S" ]]; then
        echo "Instalando dependencias..."
        pip3 install -r requirements.txt
        echo "✓ Dependencias instaladas"
    else
        echo "⚠ Puedes instalar las dependencias más tarde con:"
        echo "   pip3 install -r requirements.txt"
    fi
else
    echo "⚠ Python NO está instalado. Instálalo para usar los scripts."
fi
echo ""

# ============================================================================
# PASO 5: ESTRUCTURA DEL REPOSITORIO
# ============================================================================
echo "📋 Paso 5: Verificando estructura..."
echo ""
echo "Estructura del repositorio:"
tree -L 2 -I '__pycache__|*.pyc' || ls -R
echo ""

# ============================================================================
# PASO 6: PRIMER COMMIT
# ============================================================================
echo "📋 Paso 6: Preparando primer commit..."
git add .
git commit -m "🎉 Initial commit: Estructura completa del curso Econometría II"
echo "✓ Primer commit creado"
echo ""

# ============================================================================
# RESUMEN
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     ✓ INICIALIZACIÓN COMPLETA                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 PRÓXIMOS PASOS:"
echo ""
echo "1. Crear el repositorio en GitHub:"
echo "   https://github.com/new"
echo "   Nombre: NOTAS_DE_CLASE_ECONOMETRIA_2"
echo ""
echo "2. Subir el código:"
echo "   git push -u origin main"
echo ""
echo "3. Probar un ejemplo:"
echo "   cd capitulo01"
echo "   python3 ejemplo01_calculo_retornos.py"
echo ""
echo "4. Ver la documentación:"
echo "   cat README.md"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "¡Repositorio listo para usar! 🚀"
echo "═══════════════════════════════════════════════════════════════════"
