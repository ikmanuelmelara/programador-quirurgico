# 🏥 Criterios de Priorización CatSalut

## Marco Normativo

Este sistema implementa los criterios de priorización quirúrgica del **Servei Català de la Salut (CatSalut)**, basados en:

- **Ordre SLT/102/2015**: Terminis màxims de referència per a l'accés als procediments quirúrgics
- **AIAQS 2010**: Priorització entre procediments quirúrgics electius amb llista d'espera
- **AQUAS**: Documentación de tiempos de espera quirúrgica

---

## Tiempos Máximos Garantizados

### Cirugía Oncológica

| Prioridad | Tiempo Máximo | Descripción |
|-----------|---------------|-------------|
| **Oncológico Prioritario** | 45 días | Tumores malignos (excepto vejiga/próstata) |
| **Oncológico Estándar** | 60 días | Tumores de vejiga y próstata |

### Cirugía Cardíaca

| Prioridad | Tiempo Máximo | Descripción |
|-----------|---------------|-------------|
| **Cardíaca** | 90 días | Cirugía valvular y coronaria |

### Otros Garantizados

| Prioridad | Tiempo Máximo | Descripción |
|-----------|---------------|-------------|
| **Garantizado 180** | 180 días | Cataratas, prótesis cadera/rodilla |

---

## Tiempos de Referencia (No Garantizados)

| Prioridad | Tiempo Referencia | Descripción | Ejemplos |
|-----------|-------------------|-------------|----------|
| **P1** | 90 días | Alta prioridad | Hernias complicadas, colecistitis |
| **P2** | 180 días | Media prioridad | Hernias simples, varices |
| **P3** | 365 días | Baja prioridad | Cirugía estética reconstructiva |

---

## Factores de Priorización

Según la literatura (AIAQS 2010), los criterios incluyen:

### 1. Gravedad de la Enfermedad (30%)
- Impacto en la supervivencia
- Riesgo de progresión
- Afectación funcional

### 2. Riesgo Asociado a la Demora (25%)
- Probabilidad de complicaciones
- Deterioro clínico esperado
- Urgencia relativa

### 3. Tiempo en Lista de Espera (25%)
- Días transcurridos desde indicación
- Porcentaje del tiempo máximo consumido
- Penalización por exceder plazo

### 4. Efectividad Clínica Esperada (10%)
- Beneficio esperado de la intervención
- Probabilidad de éxito
- Evidencia científica

### 5. Impacto en Calidad de Vida (10%)
- Limitación funcional actual
- Dolor y síntomas
- Impacto social/laboral

---

## Implementación en el Sistema

### Cálculo del Score Clínico

```python
def calcular_score_clinico(solicitud):
    score = 0.0
    
    # 1. Prioridad CatSalut (30 puntos max)
    prioridad_scores = {
        'URGENTE': 30,
        'ONCOLOGICO_PRIORITARIO': 28,
        'ONCOLOGICO_ESTANDAR': 25,
        'CARDIACA': 22,
        'REFERENCIA_P1': 18,
        'REFERENCIA_P2': 12,
        'REFERENCIA_P3': 8,
    }
    score += prioridad_scores[solicitud.prioridad]
    
    # 2. Tiempo en espera relativo (25 puntos max)
    pct_tiempo = solicitud.dias_en_espera / tiempo_maximo[solicitud.prioridad]
    if pct_tiempo >= 1.0:  # Fuera de plazo
        score += 25
    else:
        score += pct_tiempo * 20
    
    # 3. Complejidad y riesgo (15 puntos max)
    score += solicitud.complejidad * 2
    if solicitud.requiere_uci:
        score += 5
    
    # 4. Riesgo del paciente ASA (15 puntos max)
    asa_score = {1: 0, 2: 3, 3: 7, 4: 12, 5: 15}
    score += asa_score[solicitud.paciente.clase_asa]
    
    # 5. Factor edad (10 puntos max)
    score += factor_riesgo_edad(solicitud.paciente.edad) * 10
    
    # 6. Comorbilidades (5 puntos max)
    score += min(len(solicitud.paciente.comorbilidades), 5)
    
    return min(score, 100)
```

### Ordenación para Programación

```python
def ordenar_por_prioridad(solicitudes):
    def score_total(s):
        base = peso_prioridad[s.prioridad]
        tiempo_bonus = s.porcentaje_tiempo_consumido * 50
        fuera_plazo_bonus = 100 if s.esta_fuera_plazo else 0
        return base + tiempo_bonus + fuera_plazo_bonus + s.score_clinico * 0.5
    
    return sorted(solicitudes, key=score_total, reverse=True)
```

---

## Reglas de Negocio Implementadas

### Regla 1: Oncológico Primera Hora
Las cirugías oncológicas complejas se programan preferentemente a primera hora de la mañana.

### Regla 2: No Superar Capacidad UCI
Máximo 4 ingresos UCI esperados por día.

### Regla 3: Respetar Tiempo de Limpieza
Mínimo 30 minutos entre cirugías (45 min si contaminada).

### Regla 4: Balance de Carga
Distribuir equitativamente la carga entre quirófanos disponibles.

### Regla 5: Especialidad-Quirófano
Respetar la asignación de especialidades a quirófanos equipados.

---

## Referencias

1. **Ordre SLT/102/2015**, de 14 de maig, per la qual s'actualitzen els terminis màxims de referència per a l'accés als procediments quirúrgics. DOGC núm. 6873.

2. **AIAQS (2010)**. Priorització entre procediments quirúrgics electius amb llista d'espera. Agència d'Informació, Avaluació i Qualitat en Salut.

3. **CatSalut**. Sistemes d'informació sanitària. Registre del conjunt mínim bàsic de dades d'activitat quirúrgica (CMBDAQ).

4. **Cardoen, B., Demeulemeester, E., & Beliën, J. (2010)**. Operating room planning and scheduling: A literature review. European Journal of Operational Research.

---

*Documentación basada en normativa vigente a febrero 2026*
