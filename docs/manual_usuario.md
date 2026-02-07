# 📖 Manual de Usuario - Programador Quirúrgico v4.9

## Índice

1. [Introducción](#introducción)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Guía de Uso](#guía-de-uso)
5. [Pestañas del Sistema](#pestañas-del-sistema)
6. [Preguntas Frecuentes](#preguntas-frecuentes)

---

## Introducción

El **Programador Quirúrgico Inteligente** es un sistema de optimización para la programación de actividad quirúrgica. Utiliza técnicas de Machine Learning, simulación Monte Carlo y algoritmos de optimización para:

- Maximizar la prioridad clínica según criterios CatSalut
- Optimizar la utilización de quirófanos
- Predecir demanda y urgencias
- Simular escenarios "What-If"

---

## Requisitos

### Para Google Colab (Recomendado)
- Cuenta de Google
- Navegador web moderno

### Para Instalación Local
- Python 3.8 o superior
- 4GB RAM mínimo
- Conexión a internet (para descargar dependencias)

---

## Instalación

### Opción 1: Google Colab

1. Sube la carpeta del proyecto a Google Drive
2. Abre el notebook `Programador_Quirurgico_v49.ipynb`
3. Ejecuta las 3 celdas en orden
4. La interfaz se abrirá automáticamente

### Opción 2: Local

```bash
pip install -r requirements.txt
python app/programador_quirurgico_colab_v49.py
```

---

## Guía de Uso

### Flujo Recomendado

```
📊 Dashboard → 📋 Lista Espera → 📈 Pred. Demanda → 🎯 Planificador → 🔮 What-If → ⚙️ Optimizar
```

1. **Dashboard**: Revisa el estado actual del bloque quirúrgico
2. **Lista Espera**: Analiza los pacientes pendientes
3. **Pred. Demanda**: Proyecta la evolución futura
4. **Planificador**: Calcula la configuración óptima de sesiones
5. **What-If**: Simula diferentes escenarios
6. **Optimizar**: Genera el programa quirúrgico final

---

## Pestañas del Sistema

### 📊 Dashboard
Muestra métricas generales:
- Total de pacientes en lista
- Pacientes fuera de plazo
- Distribución por prioridad
- Distribución por especialidad

### 📋 Lista de Espera
Tabla con los pacientes pendientes:
- ID, nombre, intervención
- Prioridad CatSalut
- Días en espera
- Score clínico

### 📈 Predicción de Demanda
Proyección de la lista de espera:
- Horizonte configurable (4-24 semanas)
- Intervalos de confianza
- Tendencias por especialidad

### 🎯 Planificador Estratégico
Análisis integral que incluye:
- Demanda actual por especialidad
- Reparto óptimo de sesiones
- Simulación del impacto
- Recomendaciones concretas

### 🔮 What-If
Simulador de escenarios:
- Añadir/quitar sesiones
- Cerrar quirófanos
- Cambiar demanda
- Comparar múltiples escenarios

### 🚨 Pred. Urgencias
Predicción de urgencias diferidas:
- Reserva sugerida por especialidad
- Predicción semanal
- Aplicar reservas ML

### 📅 Sesiones
Configuración de sesiones quirúrgicas:
- Matriz quirófano × día × turno
- Asignación de especialidades
- Aplicar configuración óptima

### 🚫 Restricciones
Constructor de restricciones manuales:
- Cirujano requiere quirófano específico
- Especialidad en días concretos
- Máximo de cirugías complejas

### ⚙️ Optimizar
Motor de optimización:
- Balance prioridad clínica / eficiencia
- Selección de método (Auto, Heurístico, Genético, MILP)
- Reservas para urgencias
- Resultados detallados

---

## Criterios de Priorización CatSalut

| Prioridad | Tiempo Máximo | Color |
|-----------|---------------|-------|
| Oncológico Prioritario | 45 días | 🔴 Rojo |
| Oncológico Estándar | 60 días | 🟠 Naranja |
| Cardíaca | 90 días | 🟣 Púrpura |
| Referencia P1 | 90 días | 🔵 Azul |
| Referencia P2 | 180 días | 🟢 Verde |
| Referencia P3 | 365 días | ⚪ Gris |

---

## Preguntas Frecuentes

### ¿Qué significa "Fuera de Plazo"?
Un paciente está fuera de plazo cuando ha superado el tiempo máximo de espera según su prioridad CatSalut.

### ¿Qué método de optimización debo usar?
- **Auto**: Recomendado. Prueba todos y selecciona el mejor.
- **Heurístico**: Rápido, bueno para pruebas.
- **Genético**: Mejor exploración, más lento.
- **MILP**: Solución exacta, requiere OR-Tools.

### ¿Cómo interpreto el Score?
- **Score Total**: Combinación ponderada de clínico y eficiencia
- **Score Clínico**: Qué tan bien se priorizan los casos urgentes
- **Score Eficiencia**: Utilización de los quirófanos

### ¿Los datos son reales?
No. El sistema genera datos sintéticos realistas para demostración. En producción, se conectaría al sistema de información hospitalario.

---

## Soporte

Para reportar problemas o sugerencias:
- Abre un Issue en GitHub
- Contacta al desarrollador

---

*Documentación actualizada: Febrero 2026*
