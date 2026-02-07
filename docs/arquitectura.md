# 🏗️ Arquitectura del Sistema

## Visión General

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTERFAZ GRADIO (app/)                           │
│  Dashboard │ Lista │ Demanda │ Planificador │ What-If │ Optimizar  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CAPA DE ORQUESTACIÓN                           │
│                          main.py                                    │
│         ProgramadorQuirurgico: coordina todos los módulos           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────┬───────────┼───────────┬──────────────┐
        ▼              ▼           ▼           ▼              ▼
┌──────────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐
│  CONSTRAINT  │ │OPTIMIZER │ │SIMULATOR│ │PREDICTOR│ │   PREDICTOR  │
│   LEARNING   │ │          │ │ WHAT-IF │ │ DEMANDA │ │  URGENCIAS   │
├──────────────┤ ├──────────┤ ├─────────┤ ├─────────┤ ├──────────────┤
│ Association  │ │Heurístico│ │  Monte  │ │ Series  │ │Random Forest │
│ Rules Mining │ │          │ │  Carlo  │ │Temporales│ │   Gradient  │
│ Clustering   │ │ Genético │ │         │ │         │ │   Boosting  │
│Decision Trees│ │  (DEAP)  │ │Erlang-C │ │ Prophet │ │             │
│Isolation For.│ │   MILP   │ │         │ │  ARIMA  │ │             │
│ Correlation  │ │(OR-Tools)│ │ Inverse │ │         │ │             │
│  Temporal    │ │          │ │  Optim. │ │         │ │             │
│ Sequential   │ │          │ │         │ │         │ │             │
└──────────────┘ └──────────┘ └─────────┘ └─────────┘ └──────────────┘
        │              │           │           │              │
        └──────────────┴───────────┼───────────┴──────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       CAPA DE DATOS                                 │
│  config.py │ models.py │ synthetic_data.py                         │
│  Configuración CatSalut │ Modelos de dominio │ Generador datos     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Módulos Principales

### 1. config.py - Configuración

**Responsabilidad**: Definir constantes y configuración del sistema.

```python
# Contenido principal:
- PrioridadCatSalut (Enum)      # Niveles de prioridad
- TIEMPOS_MAXIMOS_ESPERA        # Días máximos por prioridad
- Especialidad (Enum)            # Especialidades quirúrgicas
- TipoIntervencion (dataclass)   # Catálogo de intervenciones
- Quirofano (dataclass)          # Configuración de quirófanos
- PesosOptimizacion (dataclass)  # Pesos configurables
- RestriccionesGlobales          # Restricciones del sistema
```

### 2. models.py - Modelos de Datos

**Responsabilidad**: Definir estructuras de datos del dominio.

```python
# Clases principales:
- Paciente                  # Datos del paciente
- Cirujano                  # Datos del cirujano
- SolicitudCirugia          # Solicitud en lista de espera
- CirugiaProgramada         # Cirugía asignada a slot
- ProgramaDiario            # Programa de un día
- ProgramaPeriodo           # Programa completo
- RestriccionAprendida      # Restricción descubierta por ML
```

### 3. synthetic_data.py - Generador de Datos

**Responsabilidad**: Generar datos sintéticos realistas.

```python
# Funcionalidades:
- Generar pacientes con demografía realista
- Generar cirujanos por especialidad
- Generar lista de espera con distribución CatSalut
- Generar histórico de cirugías (365 días)
```

### 4. constraint_learning.py - Aprendizaje Básico

**Responsabilidad**: Descubrir restricciones implícitas del histórico.

```python
# Técnicas:
- Preferencias cirujano-quirófano
- Patrones de secuenciación
- Restricciones temporales
- Patrones de duración
- Asignación especialidad-quirófano
- Patrones por día de semana
```

### 5. constraint_learning_advanced.py - ML Avanzado

**Responsabilidad**: Técnicas avanzadas de Machine Learning.

```python
# 8 Técnicas:
1. Association Rules Mining (Apriori/FP-Growth)
2. Clustering K-Means
3. Clustering DBSCAN
4. Decision Trees (reglas interpretables)
5. Isolation Forest (detección anomalías)
6. Análisis de correlación multivariable
7. Patrones temporales avanzados
8. Sequential Pattern Mining
```

### 6. optimizer.py - Optimizador Básico

**Responsabilidad**: Motor de optimización heurístico.

```python
# Algoritmos:
- First Fit Decreasing (heurística constructiva)
- Hill Climbing (búsqueda local)
- Función objetivo multi-criterio
```

### 7. optimizer_advanced.py - Optimización Avanzada

**Responsabilidad**: Algoritmos de optimización avanzados.

```python
# Algoritmos:
- Algoritmo Genético (DEAP)
  - Selección por torneo
  - Cruce de dos puntos
  - Mutación adaptativa
  - Elitismo

- MILP (OR-Tools)
  - Modelo exacto
  - Variables binarias de asignación
  - Restricciones duras y blandas
```

### 8. simulador_whatif.py - Simulación

**Responsabilidad**: Simular escenarios futuros.

```python
# Componentes:
- ModeloCapacidad: cálculo determinista
- ModeloColas: teoría de colas (Erlang-C)
- SimuladorMonteCarlo: simulación estocástica
- OptimizadorInverso: configuración óptima
```

### 9. predictor_demanda.py - Predicción Demanda

**Responsabilidad**: Predecir evolución de lista de espera.

```python
# Técnicas:
- Series temporales
- Regresión con features temporales
- Proyección con intervalos de confianza
```

### 10. urgencias_predictor.py - Predicción Urgencias

**Responsabilidad**: Predecir urgencias diferidas.

```python
# Técnicas:
- Random Forest por especialidad
- Gradient Boosting
- Features: día semana, mes, estacionalidad
- Salida: reserva recomendada en minutos
```

---

## Flujo de Datos

### Inicialización

```
1. synthetic_data genera datos
2. constraint_learning analiza histórico
3. constraint_learning_advanced descubre patrones ML
4. Predictores se entrenan con histórico
5. Sistema listo para optimizar
```

### Optimización

```
1. Usuario configura pesos y método
2. optimizer recibe lista de espera
3. Aplica restricciones aprendidas
4. Ejecuta algoritmo seleccionado
5. Retorna programa optimizado
```

### Simulación What-If

```
1. Usuario define escenario
2. SimuladorMonteCarlo ejecuta N simulaciones
3. Calcula estadísticas y probabilidades
4. Retorna proyección con intervalos de confianza
```

---

## Patrones de Diseño

### Dependency Injection
Los módulos reciben sus dependencias como parámetros, facilitando testing y flexibilidad.

### Strategy Pattern
Los algoritmos de optimización implementan una interfaz común, permitiendo intercambiarlos.

### Factory Pattern
`crear_predictor_desde_historico()` encapsula la creación compleja de predictores.

### Observer Pattern
La interfaz Gradio observa cambios y actualiza visualizaciones.

---

## Extensibilidad

### Añadir nuevo algoritmo de optimización

1. Crear clase en `optimizer_advanced.py`
2. Implementar método `optimizar(solicitudes, cirujanos, ...)`
3. Registrar en selector de métodos

### Añadir nueva técnica de ML

1. Crear método en `constraint_learning_advanced.py`
2. Implementar extracción de restricciones
3. Añadir al pipeline de análisis

### Añadir nuevo predictor

1. Crear archivo `nuevo_predictor.py`
2. Implementar clase con métodos `entrenar()` y `predecir()`
3. Integrar en `main.py`

---

## Rendimiento

| Operación | Tiempo típico | Complejidad |
|-----------|---------------|-------------|
| Generación datos (250 pacientes) | ~2s | O(n) |
| Aprendizaje restricciones | ~5s | O(n²) |
| Optimización heurística | ~3s | O(n log n) |
| Optimización genética | ~15s | O(g × p × n) |
| Simulación Monte Carlo (300) | ~2s | O(s × w) |

Donde: n=pacientes, g=generaciones, p=población, s=simulaciones, w=semanas

---

*Arquitectura v4.9 - Febrero 2026*
