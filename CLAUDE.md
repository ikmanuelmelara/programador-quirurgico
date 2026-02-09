# CLAUDE.md - Programador Quirúrgico Inteligente

## 🎯 Descripción del Proyecto

Sistema de optimización para programación quirúrgica en un bloque de 8 quirófanos, basado en criterios de priorización **CatSalut** (sistema público de salud de Catalunya).

**Objetivo**: Ayudar a gestores de bloques quirúrgicos a optimizar la programación maximizando el cumplimiento de tiempos de espera garantizados mientras se optimiza la utilización de recursos.

---

## 🏗️ Arquitectura

```
src/                              # Módulos Python del backend
├── config.py                     # Configuración CatSalut, quirófanos, pesos
├── models.py                     # Modelos de datos (Paciente, Cirugía, etc.)
├── synthetic_data.py             # Generador de datos sintéticos realistas
├── constraint_learning.py        # Aprendizaje básico de restricciones
├── constraint_learning_advanced.py # ML avanzado (8 técnicas)
├── optimizer.py                  # Optimizador heurístico
├── optimizer_advanced.py         # Genético (DEAP) + MILP (OR-Tools)
├── simulador_whatif.py           # Simulación Monte Carlo, What-If
├── predictor_demanda.py          # Predicción de evolución lista espera
└── urgencias_predictor.py        # Predicción de urgencias diferidas

app/
└── programador_quirurgico_colab_v49.py  # Aplicación Gradio (interfaz completa)

notebooks/
└── Programador_Quirurgico_v49.ipynb     # Notebook para Google Colab
```

---

## 🔧 Stack Tecnológico

- **Python 3.8+**
- **Gradio**: Interfaz web
- **NumPy/Pandas**: Procesamiento de datos
- **Scikit-learn**: Machine Learning
- **DEAP**: Algoritmos genéticos (opcional)
- **OR-Tools**: Optimización MILP (opcional)
- **Plotly**: Visualizaciones interactivas

---

## 📋 Criterios CatSalut Implementados

| Prioridad | Tiempo Máximo | Código |
|-----------|---------------|--------|
| Oncológico Prioritario | 45 días | `ONCOLOGICO_PRIORITARIO` |
| Oncológico Estándar | 60 días | `ONCOLOGICO_ESTANDAR` |
| Cardíaca | 90 días | `CARDIACA` |
| Referencia P1 | 90 días | `REFERENCIA_P1` |
| Referencia P2 | 180 días | `REFERENCIA_P2` |
| Referencia P3 | 365 días | `REFERENCIA_P3` |

---

## 🚀 Comandos Útiles

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación Gradio localmente
python app/programador_quirurgico_colab_v49.py

# Ejecutar tests
pytest tests/ -v

# Ejecutar módulo principal (demo)
python src/main.py
```

---

## 📁 Archivos Clave para Modificaciones

### Para cambios en la interfaz:
- `app/programador_quirurgico_colab_v49.py` - Archivo principal Gradio (~4500 líneas)

### Para cambios en lógica de negocio:
- `src/simulador_whatif.py` - Simulación y optimización inversa
- `src/predictor_demanda.py` - Predicción de demanda
- `src/optimizer.py` / `src/optimizer_advanced.py` - Motor de optimización

### Para cambios en configuración:
- `src/config.py` - Prioridades, tiempos, quirófanos

---

## ⚠️ Conceptos Importantes

### Predicción vs Prescripción

**PREDICCIÓN** (módulo `predictor_demanda.py`):
- Responde: "¿Qué pasará si no hacemos nada?"
- Basado en: histórico, tendencias, estacionalidad
- Output: proyección con intervalos de confianza

**PRESCRIPCIÓN** (módulo `simulador_whatif.py` - `OptimizadorInverso`):
- Responde: "¿Qué debo hacer para conseguir X?"
- Basado en: objetivos del usuario + restricciones
- Output: configuración recomendada de sesiones

### Sesiones vs Cirugías
- **Sesión**: Bloque de tiempo en quirófano (mañana ~7h, tarde ~5h)
- **Cirugía**: Intervención individual dentro de una sesión

### Lista de Espera
- Pacientes pendientes de operar
- "Fuera de plazo" = han superado tiempo máximo CatSalut

---

## 🐛 Problemas Conocidos / Deuda Técnica

1. **Predicción no usa configuración de sesiones**: El predictor de demanda usa capacidad fija histórica, no la configuración actual de sesiones.

2. **Objetivo de optimización incorrecto**: El planificador actual busca "eliminar lista" en lugar de "equilibrar flujo + cumplir CatSalut".

3. **Archivo monolítico**: `programador_quirurgico_colab_v49.py` tiene ~4500 líneas. Considerar refactorizar.

---

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Test específico
pytest tests/test_sistema.py::TestOptimizer -v
```

---

## 📝 Convenciones de Código

- **Idioma código**: Inglés para nombres de funciones/variables
- **Idioma UI/docs**: Español (usuarios son profesionales sanitarios españoles)
- **Docstrings**: En español, formato descriptivo
- **Type hints**: Usar cuando sea posible
- **Formato**: PEP 8

---

## 🔗 Referencias

- [Normativa CatSalut](https://catsalut.gencat.cat/)
- [Ordre SLT/102/2015](https://dogc.gencat.cat/) - Tiempos máximos quirúrgicos
- [AIAQS 2010](https://aquas.gencat.cat/) - Criterios de priorización
