# ⚡ REFERENCIA RÁPIDA - 3 MINUTOS

## 🎯 PARA EMPEZAR AHORA MISMO:

### Opción A: Solo quiero probar los scripts (SIN GitHub)

```bash
cd NOTAS_DE_CLASE_ECONOMETRIA_2
pip install -r requirements.txt
cd capitulo01
python ejemplo01_calculo_retornos.py
```

¡Listo! Ya viste cómo funciona. ✅

---

### Opción B: Quiero subirlo a GitHub

```bash
cd NOTAS_DE_CLASE_ECONOMETRIA_2
./inicializar.sh
# Sigue las instrucciones en pantalla
```

Luego crea el repo en: https://github.com/new  
Nombre: `NOTAS_DE_CLASE_ECONOMETRIA_2`

```bash
git push -u origin main
```

¡Listo en GitHub! ✅

---

## 📝 Para LaTeX: Reemplaza en tu preámbulo

```latex
\newcommand{\repoGitHub}{%
    \url{https://github.com/JeelCueva/NOTAS_DE_CLASE_ECONOMETRIA_2}%
}
```

---

## 📁 Archivos Importantes:

- `README.md` → Documentación completa
- `GUIA_USO_INMEDIATO.md` → Guía detallada
- `capitulo01/ejemplo01_calculo_retornos.py` → Script de ejemplo
- `datos/datos_activos_financieros.xlsx` → Base de datos
- `inicializar.sh` → Script de configuración automática

---

## ⚡ Comandos Útiles:

```bash
# Ejecutar ejemplo
cd capitulo01 && python ejemplo01_calculo_retornos.py

# Subir cambios
git add . && git commit -m "Update" && git push

# Ver estructura
tree -L 2
```

---

**¡ESO ES TODO! Ya puedes usar el repositorio.** 🚀

Ver `GUIA_USO_INMEDIATO.md` para más detalles.
