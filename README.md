# 🏥 Programador Quirúrgico Inteligente v4.9

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ikmanuelmelara/programador-quirurgico/blob/main/notebooks/Programador_Quirurgico_v49.ipynb)

Sistema avanzado de **optimización para programación quirúrgica** con Machine Learning, simulación Monte Carlo y criterios de priorización del sistema público de salud de Catalunya (CatSalut).

![Dashboard Preview](docs/images/dashboard_preview.png)

## 🎯 Características Principales

### Optimización Inteligente
- **3 algoritmos de optimización**: Heurístico, Genético (DEAP) y MILP (OR-Tools)
- **Modo AUTO**: Selecciona automáticamente el mejor algoritmo
- **Balance configurable**: Prioridad clínica vs eficiencia operativa

### Machine Learning
- **8 técnicas de aprendizaje** de restricciones implícitas
- **Predicción de demanda** con Prophet/ARIMA
- **Predicción de urgencias** diferidas por especialidad
- **Detección de anomalías** con Isolation Forest

### Simulación What-If
- **Monte Carlo** con 300+ simulaciones
- **Teoría de colas** (Erlang-C)
- **Optimización inversa** para configuración óptima
- **Intervalos de confianza** al 80%

### Cumplimiento Normativo
- **Criterios CatSalut** de priorización
- **Tiempos garantizados**: Oncológico (45-60 días), Cardíaco (90 días)
- **Tiempos de referencia**: P1 (90d), P2 (180d), P3 (365d)

## 📊 Módulos del Sistema

| Módulo | Descripción | Técnicas |
|--------|-------------|----------|
| `constraint_learning` | Aprendizaje de restricciones | Association Rules, Clustering, Decision Trees |
| `optimizer` | Motor de optimización | Heurístico + Búsqueda local |
| `optimizer_advanced` | Optimización avanzada | Algoritmo Genético, MILP |
| `simulador_whatif` | Simulación de escenarios | Monte Carlo, Erlang-C |
| `predictor_demanda` | Predicción de demanda | Series temporales, ML |
| `urgencias_predictor` | Predicción de urgencias | Random Forest, Gradient Boosting |

## 🚀 Inicio Rápido

### Opción 1: Google Colab (Recomendado)

1. Haz clic en el badge "Open in Colab" arriba
2. Ejecuta las 3 celdas en orden
3. ¡Listo! La interfaz se abrirá automáticamente

### Opción 2: Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/ikmanuelmelara/programador-quirurgico.git
cd programador-quirurgico

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python -m src.main
```

### Opción 3: Docker

```bash
docker build -t programador-quirurgico .
docker run -p 7860:7860 programador-quirurgico
```

## 📁 Estructura del Proyecto

```
programador-quirurgico/
├── 📂 src/
│   ├── config.py                      # Configuración y constantes CatSalut
│   ├── models.py                      # Modelos de datos
│   ├── main.py                        # Orquestador principal
│   ├── synthetic_data.py              # Generador de datos sintéticos
│   ├── constraint_learning.py         # Aprendizaje básico de restricciones
│   ├── constraint_learning_advanced.py # ML avanzado (8 técnicas)
│   ├── optimizer.py                   # Optimizador básico
│   ├── optimizer_advanced.py          # Genético + MILP
│   ├── simulador_whatif.py            # Simulación Monte Carlo
│   ├── predictor_demanda.py           # Predicción de demanda
│   └── urgencias_predictor.py         # Predicción de urgencias
├── 📂 notebooks/
│   └── Programador_Quirurgico_v49.ipynb
├── 📂 app/
│   └── programador_quirurgico_colab_v49.py  # Aplicación Gradio
├── 📂 docs/
│   ├── manual_usuario.md
│   ├── arquitectura.md
│   └── images/
├── 📂 tests/
│   └── test_optimizer.py
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md
```

## 🔧 Configuración

### Pesos de Optimización

```python
# En la interfaz o por código
peso_clinico = 0.6      # 60% prioridad clínica
peso_eficiencia = 0.4   # 40% eficiencia operativa
```

### Criterios de Priorización CatSalut

| Prioridad | Tiempo Máximo | Descripción |
|-----------|---------------|-------------|
| Oncológico Prioritario | 45 días | Tumores malignos |
| Oncológico Estándar | 60 días | Vejiga, próstata |
| Cardíaca | 90 días | Valvular, coronaria |
| Referencia P1 | 90 días | Alta prioridad |
| Referencia P2 | 180 días | Media prioridad |
| Referencia P3 | 365 días | Baja prioridad |

## 📈 Ejemplo de Uso

```python
from src.main import ProgramadorQuirurgico

# Inicializar
programador = ProgramadorQuirurgico(seed=42)
programador.inicializar_datos_sinteticos(n_solicitudes=250)

# Configurar balance
programador.configurar_pesos(peso_clinico=0.6, peso_eficiencia=0.4)

# Optimizar
resultado = programador.optimizar_programa(horizonte_dias=14)

# Ver resultados
print(f"Score: {resultado.score_total:.4f}")
print(f"Cirugías programadas: {resultado.cirugias_programadas}")
print(f"Utilización: {resultado.score_eficiencia:.1%}")
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📚 Documentación

- [Manual de Usuario](docs/manual_usuario.md)
- [Arquitectura del Sistema](docs/arquitectura.md)
- [API Reference](docs/api.md)
- [Criterios CatSalut](docs/catsalut.md)

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **CatSalut** - Criterios de priorización quirúrgica
- **AIAQS/AQUAS** - Documentación de tiempos de espera
- Literatura científica en OR para healthcare scheduling

## 📧 Contacto

**Manuel Melara Otamendi** - ik.manuel.melara@gmail.com

Link del proyecto: [https://github.com/TU_USUARIO/programador-quirurgico](https://github.com/ikmanuelmelara/programador-quirurgico)

---

⭐ Si este proyecto te resulta útil, ¡considera darle una estrella!
