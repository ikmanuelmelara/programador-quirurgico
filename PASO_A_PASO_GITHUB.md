# 📚 Guía Paso a Paso: Subir el Proyecto a GitHub

## 📋 Requisitos Previos

1. **Cuenta de GitHub**: Si no tienes una, créala en [github.com](https://github.com)
2. **Git instalado**: Descarga de [git-scm.com](https://git-scm.com/downloads)
3. **Los archivos del proyecto** (este ZIP)

---

## 🚀 Paso 1: Crear el Repositorio en GitHub

1. Ve a [github.com](https://github.com) e inicia sesión
2. Haz clic en el botón verde **"New"** o el **"+"** en la esquina superior derecha
3. Rellena los datos:
   - **Repository name**: `programador-quirurgico`
   - **Description**: `Sistema de optimización para programación quirúrgica con ML`
   - **Visibility**: Public (o Private si prefieres)
   - ⚠️ **NO marques** "Add a README file" (ya tenemos uno)
   - ⚠️ **NO marques** "Add .gitignore" (ya tenemos uno)
   - **License**: None (ya tenemos LICENSE)
4. Clic en **"Create repository"**

---

## 🚀 Paso 2: Preparar los Archivos Localmente

### Opción A: Desde Windows (con Git Bash o CMD)

```bash
# 1. Crear carpeta del proyecto
mkdir programador-quirurgico
cd programador-quirurgico

# 2. Descomprimir el ZIP aquí (hazlo manualmente o con unzip)
# Asegúrate de que la estructura quede así:
#   programador-quirurgico/
#   ├── src/
#   ├── app/
#   ├── notebooks/
#   ├── README.md
#   ├── requirements.txt
#   └── ...
```

### Opción B: Desde Mac/Linux

```bash
# 1. Crear carpeta y descomprimir
mkdir programador-quirurgico
cd programador-quirurgico
unzip ~/Downloads/programador_quirurgico_github.zip -d .
```

---

## 🚀 Paso 3: Inicializar Git y Subir

Abre una terminal en la carpeta del proyecto:

```bash
# 1. Inicializar repositorio Git
git init

# 2. Configurar tu identidad (solo la primera vez)
git config user.name "Tu Nombre"
git config user.email "tu@email.com"

# 3. Añadir todos los archivos
git add .

# 4. Crear el primer commit
git commit -m "🎉 Initial commit: Programador Quirúrgico v4.9"

# 5. Renombrar la rama principal a 'main'
git branch -M main

# 6. Conectar con GitHub (REEMPLAZA TU_USUARIO)
git remote add origin https://github.com/TU_USUARIO/programador-quirurgico.git

# 7. Subir al repositorio
git push -u origin main
```

### Si te pide autenticación:

GitHub ya no acepta contraseñas. Necesitas un **Personal Access Token**:

1. Ve a GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Clic en "Generate new token (classic)"
3. Nombre: "Git CLI"
4. Selecciona: `repo` (todos los permisos de repo)
5. Clic en "Generate token"
6. **COPIA EL TOKEN** (solo se muestra una vez)
7. Cuando Git te pida password, pega el token

---

## 🚀 Paso 4: Verificar

1. Ve a `https://github.com/TU_USUARIO/programador-quirurgico`
2. Deberías ver todos tus archivos
3. El README.md se mostrará automáticamente

---

## 🚀 Paso 5: Configurar el Badge de Colab (Opcional)

Edita el README.md y reemplaza `TU_USUARIO` con tu nombre de usuario real de GitHub en esta línea:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/programador-quirurgico/blob/main/notebooks/Programador_Quirurgico_v49.ipynb)
```

---

## 📁 Estructura Final en GitHub

```
programador-quirurgico/
├── 📂 src/                           # Código fuente
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── main.py
│   ├── synthetic_data.py
│   ├── constraint_learning.py
│   ├── constraint_learning_advanced.py
│   ├── optimizer.py
│   ├── optimizer_advanced.py
│   ├── simulador_whatif.py
│   ├── predictor_demanda.py
│   └── urgencias_predictor.py
├── 📂 app/                           # Aplicación Gradio
│   └── programador_quirurgico_colab_v49.py
├── 📂 notebooks/                     # Notebooks
│   └── Programador_Quirurgico_v49.ipynb
├── 📂 docs/                          # Documentación
│   └── (opcional)
├── 📂 tests/                         # Tests
│   └── (opcional)
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🔄 Comandos Útiles para el Futuro

### Actualizar el repositorio con cambios:

```bash
git add .
git commit -m "Descripción del cambio"
git push
```

### Descargar cambios del repositorio:

```bash
git pull
```

### Ver estado:

```bash
git status
```

### Ver historial:

```bash
git log --oneline
```

---

## 🎯 Crear un Release (Versión)

1. En GitHub, ve a tu repositorio
2. Clic en "Releases" (columna derecha)
3. Clic en "Create a new release"
4. Tag: `v4.9.0`
5. Title: `Programador Quirúrgico v4.9`
6. Descripción: Lista de características
7. Adjunta el ZIP si quieres
8. Clic en "Publish release"

---

## ❓ Problemas Comunes

### "fatal: not a git repository"
→ Asegúrate de estar en la carpeta correcta y haber ejecutado `git init`

### "Permission denied"
→ Usa el Personal Access Token en lugar de tu contraseña

### "remote origin already exists"
→ Ejecuta: `git remote remove origin` y vuelve a añadirlo

### Los archivos no aparecen en GitHub
→ Verifica que hiciste `git add .` y `git commit` antes de `git push`

---

## 🎉 ¡Listo!

Tu proyecto está ahora en GitHub. Puedes:
- Compartir el enlace
- Activar GitHub Pages para documentación
- Configurar GitHub Actions para CI/CD
- Invitar colaboradores

**URL de tu repositorio**: `https://github.com/TU_USUARIO/programador-quirurgico`
