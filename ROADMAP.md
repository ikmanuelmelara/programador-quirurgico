# 🗺️ ROADMAP - Programador Quirúrgico Inteligente

## Versión Actual: 4.9

---

# 🎯 EVOLUTIVO PRIORITARIO: Separación Predicción/Prescripción

## Contexto del Problema

Actualmente el sistema mezcla conceptos de **predicción** (¿qué pasará?) con **prescripción** (¿qué debo hacer?), lo que genera confusión y resultados poco útiles.

### Problema específico en el Planificador Estratégico:

**Comportamiento actual (INCORRECTO):**
```
- Calcula "sesiones óptimas" para OPERAR A TODOS los pacientes en lista
- Resultado: recomienda ~100 sesiones → lista baja a 0
- Esto NO es realista ni es lo que necesita un gestor
```

**Comportamiento deseado (CORRECTO):**
```
- El usuario DEFINE sus objetivos (ej: "cero fuera de plazo en 12 semanas")
- El sistema CALCULA la configuración mínima para lograr esos objetivos
- Resultado: recomienda +3 sesiones → cumple CatSalut, lista estable
```

---

## Especificación Funcional

### 1. Nueva Estructura de Pestañas

```
ANTES:
├── Dashboard
├── Lista Espera  
├── Pred. Demanda      ← Mezcla predicción con prescripción
├── Planificador       ← Objetivo incorrecto (eliminar lista)
├── What-If
├── Pred. Urgencias
├── Sesiones
├── Restricciones
└── Optimizar

DESPUÉS:
├── Dashboard
├── Lista Espera
├── 📊 Predicción       ← NUEVA: agrupa predicciones
│   ├── Demanda (lista de espera)
│   └── Urgencias
├── 💊 Prescripción     ← NUEVA: objetivos + optimización
│   ├── Definir Objetivos
│   ├── Calcular Configuración
│   └── Comparar Escenarios
├── 🔮 What-If          ← Se mantiene (simulación manual)
├── Sesiones
├── Restricciones
└── Optimizar (programa diario)
```

### 2. Pestaña Prescripción - Interfaz

```
┌─────────────────────────────────────────────────────────────┐
│  💊 PRESCRIPCIÓN - Configuración Óptima de Sesiones         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  🎯 DEFINE TUS OBJETIVOS                                    │
│  ─────────────────────────────────────────                  │
│                                                              │
│  Cumplimiento CatSalut:                                     │
│    ☑️ Cero pacientes fuera de plazo al final del horizonte  │
│    ☐ Reducir fuera de plazo un [__]% en el horizonte        │
│    ☑️ Oncológicos siempre dentro de plazo (45/60 días)      │
│                                                              │
│  Gestión de Lista:                                          │
│    ☐ Mantener lista estable (equilibrar flujo)              │
│    ☐ Reducir lista total un [__]%                           │
│    ☐ No superar [___] pacientes en lista                    │
│                                                              │
│  Eficiencia:                                                │
│    ☑️ Minimizar sesiones adicionales necesarias             │
│    ☐ Utilización mínima de quirófanos: [85]%                │
│                                                              │
│  ⏱️ Horizonte: [12 ▼] semanas                               │
│                                                              │
│  [🔍 Calcular Configuración Óptima]                         │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📋 RESULTADO                                                │
│  ─────────────────────────────────────────                  │
│                                                              │
│  ✅ Se encontró configuración que cumple los objetivos      │
│                                                              │
│  | Especialidad  | Actual | Recomendado | Δ    |            │
│  |---------------|--------|-------------|------|            │
│  | Colorrectal   | 5      | 8           | +3   |            │
│  | Mama          | 5      | 6           | +1   |            │
│  | Digestiva     | 10     | 10          | 0    |            │
│  | ...           |        |             |      |            │
│                                                              │
│  📈 Proyección con esta configuración:                      │
│  | Métrica        | Actual | Semana 12 |                    │
│  |----------------|--------|-----------|                    │
│  | Lista espera   | 500    | 485       |                    │
│  | Fuera de plazo | 50     | 0 ✅      |                    │
│                                                              │
│  [📥 Aplicar a Sesiones] [💾 Exportar]                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3. Lógica de Optimización (Backend)

**Archivo**: `src/simulador_whatif.py` - Clase `OptimizadorInverso`

```python
class ObjetivoPrescripcion:
    """Objetivos definidos por el usuario"""
    # Cumplimiento CatSalut
    cero_fuera_plazo: bool = True
    reducir_fp_porcentaje: Optional[float] = None  # ej: 0.5 = reducir 50%
    oncologicos_siempre_en_plazo: bool = True
    
    # Gestión de lista
    lista_estable: bool = False  # equilibrar flujo
    reducir_lista_porcentaje: Optional[float] = None
    lista_maxima: Optional[int] = None
    
    # Eficiencia
    minimizar_sesiones: bool = True
    utilizacion_minima: float = 0.85
    
    # Horizonte
    semanas: int = 12


class OptimizadorPrescriptivo:
    """
    Encuentra la configuración MÍNIMA de sesiones para cumplir objetivos.
    
    Algoritmo:
    1. Calcular flujo de entrada esperado (del predictor de demanda)
    2. Para cada objetivo activo, calcular restricciones
    3. Buscar configuración mínima que satisfaga todas las restricciones
    4. Simular con Monte Carlo para validar probabilidad de éxito
    """
    
    def optimizar(self, objetivos: ObjetivoPrescripcion) -> ResultadoPrescripcion:
        # 1. Obtener predicción de demanda (entradas esperadas)
        flujo_entrada = self.predictor.obtener_flujo_semanal()
        
        # 2. Calcular capacidad necesaria por objetivo
        if objetivos.cero_fuera_plazo:
            capacidad_fp = self._calcular_capacidad_para_cero_fp()
        
        if objetivos.lista_estable:
            capacidad_equilibrio = self._calcular_capacidad_equilibrio(flujo_entrada)
        
        # 3. Buscar configuración mínima
        config_optima = self._buscar_minimo(...)
        
        # 4. Validar con simulación
        resultado_sim = self.simulador.simular(config_optima)
        
        return ResultadoPrescripcion(
            configuracion=config_optima,
            prob_exito=resultado_sim.prob_cumplir_objetivos,
            proyeccion=resultado_sim
        )
```

### 4. Conexión Predicción → Prescripción

**Crítico**: La prescripción debe usar el **flujo de entrada** del predictor, no la lista acumulada.

```python
# INCORRECTO (actual):
horas_necesarias = sum(paciente.duracion for paciente in lista_espera)  # TODA la lista

# CORRECTO (nuevo):
flujo_semanal = predictor.obtener_entradas_esperadas_semana()  # ~85 pacientes
horas_equilibrio = flujo_semanal * duracion_media  # Para mantener estable

horas_reducir_fp = pacientes_fuera_plazo * duracion_media / semanas_horizonte  # Gradual

horas_necesarias = horas_equilibrio + horas_reducir_fp
```

---

## Archivos a Modificar

| Archivo | Cambios |
|---------|---------|
| `app/programador_quirurgico_colab_v49.py` | Nueva pestaña Prescripción, reorganizar pestañas |
| `src/simulador_whatif.py` | Nueva clase `OptimizadorPrescriptivo`, refactorizar `OptimizadorInverso` |
| `src/predictor_demanda.py` | Añadir método `obtener_flujo_semanal()` |

---

## Criterios de Aceptación

- [ ] Usuario puede seleccionar objetivos de una lista de checkboxes
- [ ] El sistema calcula configuración MÍNIMA (no máxima) para cumplir objetivos
- [ ] La proyección muestra lista estabilizada, NO bajando a cero (salvo que se pida)
- [ ] Se muestra probabilidad de éxito (del Monte Carlo)
- [ ] Botón "Aplicar a Sesiones" traslada la configuración recomendada
- [ ] Separación clara visual entre Predicción y Prescripción

---

# 📋 OTROS EVOLUTIVOS PENDIENTES

## Prioridad Alta

### E2: Predicción conectada a configuración de sesiones
**Estado**: Pendiente  
**Problema**: El predictor usa capacidad fija histórica, no la configuración actual.  
**Solución**: Que `predictor_demanda.py` reciba `configuracion_sesiones` como parámetro.

### E3: Mejorar generación de datos sintéticos
**Estado**: Pendiente  
**Problema**: Los datos sintéticos no reflejan bien la estacionalidad real.  
**Solución**: Añadir más patrones realistas (vacaciones, festivos, epidemias).

---

## Prioridad Media

### E4: Refactorizar archivo monolítico
**Estado**: Pendiente  
**Problema**: `programador_quirurgico_colab_v49.py` tiene ~4500 líneas.  
**Solución**: Separar en módulos (ui_dashboard.py, ui_prediccion.py, etc.)

### E5: Persistencia de configuración
**Estado**: Pendiente  
**Problema**: La configuración se pierde al recargar.  
**Solución**: Guardar/cargar configuración en JSON.

### E6: Exportación de informes
**Estado**: Pendiente  
**Mejora**: Exportar informes en PDF/Excel con gráficos.

---

## Prioridad Baja

### E7: Multi-idioma
**Estado**: Pendiente  
**Mejora**: Soporte para catalán además de español.

### E8: Integración con HIS
**Estado**: Futuro  
**Mejora**: Conectores para sistemas de información hospitalaria reales.

---

# 📅 Plan de Versiones

| Versión | Contenido | Estado |
|---------|-----------|--------|
| 4.9 | Versión actual estable | ✅ Completada |
| 5.0 | Separación Predicción/Prescripción (E1) | 🔄 En desarrollo |
| 5.1 | Predicción conectada a sesiones (E2) | Pendiente |
| 5.2 | Refactorización + Persistencia (E4, E5) | Pendiente |
| 6.0 | Exportación avanzada + mejoras UX | Pendiente |
