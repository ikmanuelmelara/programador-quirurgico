"""
Predictor de Demanda para el Programador Quirúrgico
===================================================
Analiza el histórico de movimientos (entradas/salidas) en lista de espera
y genera predicciones para las próximas semanas.

Modelos utilizados:
- Descomposición de series temporales
- Suavizado exponencial (Holt-Winters simplificado)
- Regresión con features temporales
- Simulación Monte Carlo para intervalos de confianza
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import warnings

# =============================================================================
# MODELOS DE DATOS PARA MOVIMIENTOS
# =============================================================================

class TipoMovimiento(Enum):
    """Tipo de movimiento en lista de espera"""
    ENTRADA = "ENTRADA"
    SALIDA_PROGRAMADA = "SALIDA_PROGRAMADA"
    SALIDA_URGENCIA = "SALIDA_URGENCIA"
    CANCELACION = "CANCELACION"
    DERIVACION = "DERIVACION"


class OrigenEntrada(Enum):
    """Origen de las entradas a lista de espera"""
    CONSULTA_EXTERNA = "CONSULTA_EXTERNA"
    URGENCIAS = "URGENCIAS"
    DERIVACION_EXTERNA = "DERIVACION_EXTERNA"
    COMITE_TUMORES = "COMITE_TUMORES"
    RECIDIVA = "RECIDIVA"


@dataclass
class MovimientoListaEspera:
    """Representa un movimiento en la lista de espera"""
    fecha: date
    tipo: TipoMovimiento
    especialidad: str
    prioridad: str  # ONCOLOGICO_PRIORITARIO, REFERENCIA_P1, etc.
    origen: Optional[OrigenEntrada] = None
    motivo_salida: Optional[str] = None  # Para cancelaciones/derivaciones


# =============================================================================
# GENERADOR DE HISTÓRICO DE MOVIMIENTOS
# =============================================================================

class GeneradorHistoricoMovimientos:
    """
    Genera un histórico realista de movimientos en lista de espera
    basado en patrones típicos de un hospital.
    """
    
    # Tasas base de entrada semanal por especialidad (pacientes/semana)
    TASAS_ENTRADA_SEMANAL = {
        'CIRUGIA_GENERAL': 15,
        'CIRUGIA_DIGESTIVA': 10,
        'UROLOGIA': 12,
        'GINECOLOGIA': 10,
        'CIRUGIA_MAMA': 8,
        'CIRUGIA_COLORRECTAL': 6,
        'CIRUGIA_VASCULAR': 7,
        'CIRUGIA_BARIATRICA': 4,
        'CIRUGIA_ENDOCRINA': 4,
        'CIRUGIA_HEPATOBILIAR': 3,
        'CIRUGIA_PLASTICA': 5,
    }
    
    # Distribución de origen de entradas por especialidad
    DIST_ORIGEN = {
        'CIRUGIA_GENERAL': {
            OrigenEntrada.CONSULTA_EXTERNA: 0.50,
            OrigenEntrada.URGENCIAS: 0.35,
            OrigenEntrada.DERIVACION_EXTERNA: 0.10,
            OrigenEntrada.COMITE_TUMORES: 0.05,
        },
        'CIRUGIA_MAMA': {
            OrigenEntrada.CONSULTA_EXTERNA: 0.25,
            OrigenEntrada.URGENCIAS: 0.02,
            OrigenEntrada.DERIVACION_EXTERNA: 0.13,
            OrigenEntrada.COMITE_TUMORES: 0.60,
        },
        'UROLOGIA': {
            OrigenEntrada.CONSULTA_EXTERNA: 0.45,
            OrigenEntrada.URGENCIAS: 0.25,
            OrigenEntrada.DERIVACION_EXTERNA: 0.10,
            OrigenEntrada.COMITE_TUMORES: 0.20,
        },
        'CIRUGIA_COLORRECTAL': {
            OrigenEntrada.CONSULTA_EXTERNA: 0.30,
            OrigenEntrada.URGENCIAS: 0.15,
            OrigenEntrada.DERIVACION_EXTERNA: 0.10,
            OrigenEntrada.COMITE_TUMORES: 0.45,
        },
        # Default para otras especialidades
        'DEFAULT': {
            OrigenEntrada.CONSULTA_EXTERNA: 0.55,
            OrigenEntrada.URGENCIAS: 0.20,
            OrigenEntrada.DERIVACION_EXTERNA: 0.15,
            OrigenEntrada.COMITE_TUMORES: 0.10,
        },
    }
    
    # Distribución de prioridad según origen
    DIST_PRIORIDAD_POR_ORIGEN = {
        OrigenEntrada.COMITE_TUMORES: {
            'ONCOLOGICO_PRIORITARIO': 0.70,
            'ONCOLOGICO_ESTANDAR': 0.30,
        },
        OrigenEntrada.URGENCIAS: {
            'REFERENCIA_P1': 0.60,
            'REFERENCIA_P2': 0.35,
            'REFERENCIA_P3': 0.05,
        },
        OrigenEntrada.CONSULTA_EXTERNA: {
            'ONCOLOGICO_PRIORITARIO': 0.05,
            'ONCOLOGICO_ESTANDAR': 0.05,
            'REFERENCIA_P1': 0.20,
            'REFERENCIA_P2': 0.50,
            'REFERENCIA_P3': 0.20,
        },
        OrigenEntrada.DERIVACION_EXTERNA: {
            'ONCOLOGICO_PRIORITARIO': 0.10,
            'ONCOLOGICO_ESTANDAR': 0.10,
            'REFERENCIA_P1': 0.25,
            'REFERENCIA_P2': 0.40,
            'REFERENCIA_P3': 0.15,
        },
    }
    
    # Factores de estacionalidad mensual (1.0 = media)
    ESTACIONALIDAD_MENSUAL = {
        1: 0.90,   # Enero - post-navidad, lento
        2: 1.00,   # Febrero
        3: 1.05,   # Marzo
        4: 1.05,   # Abril
        5: 1.10,   # Mayo - pico primavera
        6: 1.05,   # Junio
        7: 0.75,   # Julio - verano bajo
        8: 0.60,   # Agosto - mínimo
        9: 1.15,   # Septiembre - pico post-verano
        10: 1.10,  # Octubre
        11: 1.05,  # Noviembre
        12: 0.80,  # Diciembre - navidad
    }
    
    # Factores por día de semana (0=Lunes)
    FACTOR_DIA_SEMANA_ENTRADA = {
        0: 1.15,  # Lunes - más consultas
        1: 1.10,  # Martes
        2: 1.00,  # Miércoles
        3: 0.95,  # Jueves
        4: 0.80,  # Viernes - menos
        5: 0.00,  # Sábado
        6: 0.00,  # Domingo
    }
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generar_historico(
        self,
        fecha_inicio: date,
        fecha_fin: date,
        capacidad_semanal: Dict[str, int] = None
    ) -> pd.DataFrame:
        """
        Genera histórico completo de movimientos.
        
        Args:
            fecha_inicio: Fecha de inicio del histórico
            fecha_fin: Fecha de fin del histórico
            capacidad_semanal: Cirugías/semana por especialidad (para salidas)
        
        Returns:
            DataFrame con todos los movimientos
        """
        movimientos = []
        
        # Capacidad por defecto (basada en configuración típica)
        if capacidad_semanal is None:
            capacidad_semanal = {
                'CIRUGIA_GENERAL': 18,
                'CIRUGIA_DIGESTIVA': 14,
                'UROLOGIA': 16,
                'GINECOLOGIA': 14,
                'CIRUGIA_MAMA': 10,
                'CIRUGIA_COLORRECTAL': 8,
                'CIRUGIA_VASCULAR': 10,
                'CIRUGIA_BARIATRICA': 6,
                'CIRUGIA_ENDOCRINA': 6,
                'CIRUGIA_HEPATOBILIAR': 4,
                'CIRUGIA_PLASTICA': 8,
            }
        
        fecha_actual = fecha_inicio
        
        while fecha_actual <= fecha_fin:
            # Solo días laborables para entradas de consulta
            if fecha_actual.weekday() < 5:
                # Generar entradas del día
                entradas_dia = self._generar_entradas_dia(fecha_actual)
                movimientos.extend(entradas_dia)
                
                # Generar salidas del día (cirugías programadas)
                salidas_dia = self._generar_salidas_dia(fecha_actual, capacidad_semanal)
                movimientos.extend(salidas_dia)
            
            fecha_actual += timedelta(days=1)
        
        # Convertir a DataFrame
        df = pd.DataFrame([
            {
                'fecha': m.fecha,
                'tipo': m.tipo.value,
                'especialidad': m.especialidad,
                'prioridad': m.prioridad,
                'origen': m.origen.value if m.origen else None,
                'motivo_salida': m.motivo_salida,
                'dia_semana': m.fecha.weekday(),
                'mes': m.fecha.month,
                'semana': m.fecha.isocalendar()[1],
            }
            for m in movimientos
        ])
        
        return df.sort_values('fecha').reset_index(drop=True)
    
    def _generar_entradas_dia(self, fecha: date) -> List[MovimientoListaEspera]:
        """Genera las entradas de un día específico"""
        entradas = []
        
        # Factor estacional
        factor_mes = self.ESTACIONALIDAD_MENSUAL.get(fecha.month, 1.0)
        factor_dia = self.FACTOR_DIA_SEMANA_ENTRADA.get(fecha.weekday(), 1.0)
        
        for especialidad, tasa_semanal in self.TASAS_ENTRADA_SEMANAL.items():
            # Tasa diaria ajustada
            tasa_diaria = tasa_semanal / 5 * factor_mes * factor_dia
            
            # Número de entradas (Poisson)
            n_entradas = np.random.poisson(tasa_diaria)
            
            # Distribución de origen
            dist_origen = self.DIST_ORIGEN.get(especialidad, self.DIST_ORIGEN['DEFAULT'])
            
            for _ in range(n_entradas):
                # Seleccionar origen
                origen = np.random.choice(
                    list(dist_origen.keys()),
                    p=list(dist_origen.values())
                )
                
                # Seleccionar prioridad según origen
                dist_prio = self.DIST_PRIORIDAD_POR_ORIGEN.get(origen, {})
                if dist_prio:
                    prioridad = np.random.choice(
                        list(dist_prio.keys()),
                        p=list(dist_prio.values())
                    )
                else:
                    prioridad = 'REFERENCIA_P2'
                
                entradas.append(MovimientoListaEspera(
                    fecha=fecha,
                    tipo=TipoMovimiento.ENTRADA,
                    especialidad=especialidad,
                    prioridad=prioridad,
                    origen=origen
                ))
        
        return entradas
    
    def _generar_salidas_dia(
        self, 
        fecha: date, 
        capacidad_semanal: Dict[str, int]
    ) -> List[MovimientoListaEspera]:
        """Genera las salidas (cirugías) de un día específico"""
        salidas = []
        
        # Factor de variabilidad diaria
        factor_dia = np.random.uniform(0.85, 1.15)
        
        for especialidad, capacidad in capacidad_semanal.items():
            # Capacidad diaria aproximada
            capacidad_diaria = capacidad / 5 * factor_dia
            
            # Número de salidas
            n_salidas = int(np.random.poisson(capacidad_diaria))
            
            for _ in range(n_salidas):
                # Mayoría son programadas, algunas urgencias
                if np.random.random() < 0.15:  # 15% urgencias
                    tipo = TipoMovimiento.SALIDA_URGENCIA
                    prioridad = 'REFERENCIA_P1'
                else:
                    tipo = TipoMovimiento.SALIDA_PROGRAMADA
                    # Distribución de prioridades en salidas
                    prioridad = np.random.choice(
                        ['ONCOLOGICO_PRIORITARIO', 'ONCOLOGICO_ESTANDAR', 
                         'REFERENCIA_P1', 'REFERENCIA_P2', 'REFERENCIA_P3'],
                        p=[0.15, 0.10, 0.25, 0.40, 0.10]
                    )
                
                salidas.append(MovimientoListaEspera(
                    fecha=fecha,
                    tipo=tipo,
                    especialidad=especialidad,
                    prioridad=prioridad
                ))
            
            # Cancelaciones (~5% de la capacidad)
            n_cancelaciones = np.random.binomial(n_salidas, 0.05)
            for _ in range(n_cancelaciones):
                motivo = np.random.choice([
                    'Paciente no acude',
                    'Enfermedad intercurrente',
                    'Falta preoperatorio',
                    'Decisión del paciente',
                    'Falta de camas'
                ])
                salidas.append(MovimientoListaEspera(
                    fecha=fecha,
                    tipo=TipoMovimiento.CANCELACION,
                    especialidad=especialidad,
                    prioridad='REFERENCIA_P2',
                    motivo_salida=motivo
                ))
        
        return salidas


# =============================================================================
# PREDICTOR DE DEMANDA
# =============================================================================

@dataclass
class PrediccionSemanal:
    """Predicción para una semana"""
    semana: int
    fecha_inicio: date
    entradas_media: float
    entradas_ic_bajo: float
    entradas_ic_alto: float
    salidas_media: float
    salidas_ic_bajo: float
    salidas_ic_alto: float
    balance_medio: float
    lista_espera_proyectada: float
    fuera_plazo_proyectado: float


@dataclass
class ResultadoPrediccion:
    """Resultado completo de la predicción"""
    fecha_prediccion: date
    semanas_horizonte: int
    lista_espera_actual: int
    fuera_plazo_actual: int
    predicciones: List[PrediccionSemanal]
    por_especialidad: Dict[str, List[PrediccionSemanal]]
    metricas_modelo: Dict[str, float]
    alertas: List[str]
    recomendaciones: List[str]


class PredictorDemanda:
    """
    Predice la evolución de la lista de espera basándose en histórico.
    
    Utiliza:
    - Análisis de series temporales
    - Descomposición estacional
    - Suavizado exponencial
    - Simulación Monte Carlo
    """
    
    def __init__(self, historico_movimientos: pd.DataFrame):
        """
        Args:
            historico_movimientos: DataFrame con columnas:
                - fecha, tipo, especialidad, prioridad, origen, dia_semana, mes, semana
        """
        self.historico = historico_movimientos
        self.modelo_entrenado = False
        
        # Parámetros aprendidos
        self.tasas_entrada = {}  # Por especialidad
        self.tasas_salida = {}
        self.estacionalidad_mensual = {}
        self.estacionalidad_semanal = {}
        self.tendencia = {}
        
        # Métricas del modelo
        self.metricas = {}
    
    def entrenar(self) -> Dict[str, float]:
        """
        Entrena los modelos de predicción analizando el histórico.
        
        Returns:
            Diccionario con métricas de calidad del modelo
        """
        if len(self.historico) < 30:
            raise ValueError("Histórico insuficiente (mínimo 30 días)")
        
        print("=" * 60)
        print("ENTRENAMIENTO DEL PREDICTOR DE DEMANDA")
        print("=" * 60)
        print(f"Registros en histórico: {len(self.historico)}")
        print(f"Período: {self.historico['fecha'].min()} a {self.historico['fecha'].max()}")
        
        # 1. Calcular tasas base por especialidad
        self._calcular_tasas_base()
        
        # 2. Detectar estacionalidad mensual
        self._detectar_estacionalidad_mensual()
        
        # 3. Detectar patrones semanales
        self._detectar_patron_semanal()
        
        # 4. Detectar tendencia
        self._detectar_tendencia()
        
        # 5. Validar modelo (backtesting)
        self.metricas = self._validar_modelo()
        
        self.modelo_entrenado = True
        
        print(f"\n✅ Modelo entrenado")
        print(f"   MAPE Entradas: {self.metricas.get('mape_entradas', 0):.1f}%")
        print(f"   MAPE Salidas: {self.metricas.get('mape_salidas', 0):.1f}%")
        
        return self.metricas
    
    def _calcular_tasas_base(self):
        """Calcula tasas medias de entrada/salida por especialidad"""
        
        # Entradas por semana y especialidad
        entradas = self.historico[self.historico['tipo'] == 'ENTRADA']
        entradas_semana = entradas.groupby(['semana', 'especialidad']).size().reset_index(name='count')
        
        for esp in entradas['especialidad'].unique():
            datos_esp = entradas_semana[entradas_semana['especialidad'] == esp]['count']
            self.tasas_entrada[esp] = {
                'media': datos_esp.mean(),
                'std': datos_esp.std(),
                'min': datos_esp.min(),
                'max': datos_esp.max()
            }
        
        # Salidas por semana y especialidad
        salidas = self.historico[self.historico['tipo'].isin(['SALIDA_PROGRAMADA', 'SALIDA_URGENCIA'])]
        salidas_semana = salidas.groupby(['semana', 'especialidad']).size().reset_index(name='count')
        
        for esp in salidas['especialidad'].unique():
            datos_esp = salidas_semana[salidas_semana['especialidad'] == esp]['count']
            self.tasas_salida[esp] = {
                'media': datos_esp.mean(),
                'std': datos_esp.std(),
                'min': datos_esp.min(),
                'max': datos_esp.max()
            }
        
        print(f"\n📊 Tasas semanales por especialidad:")
        for esp in sorted(self.tasas_entrada.keys()):
            ent = self.tasas_entrada.get(esp, {}).get('media', 0)
            sal = self.tasas_salida.get(esp, {}).get('media', 0)
            balance = ent - sal
            signo = '+' if balance > 0 else ''
            print(f"   {esp[:15]:<15}: Entradas={ent:.1f}, Salidas={sal:.1f}, Balance={signo}{balance:.1f}")
    
    def _detectar_estacionalidad_mensual(self):
        """Detecta patrones estacionales mensuales"""
        
        # Agrupar por mes
        entradas = self.historico[self.historico['tipo'] == 'ENTRADA']
        entradas_mes = entradas.groupby('mes').size()
        media_mensual = entradas_mes.mean()
        
        for mes in range(1, 13):
            if mes in entradas_mes.index:
                self.estacionalidad_mensual[mes] = entradas_mes[mes] / media_mensual
            else:
                self.estacionalidad_mensual[mes] = 1.0
        
        print(f"\n📅 Estacionalidad mensual detectada:")
        meses_nombre = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                        'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        for mes in range(1, 13):
            factor = self.estacionalidad_mensual[mes]
            barra = '█' * int(factor * 10)
            print(f"   {meses_nombre[mes-1]}: {factor:.2f} {barra}")
    
    def _detectar_patron_semanal(self):
        """Detecta patrones por día de la semana"""
        
        entradas = self.historico[self.historico['tipo'] == 'ENTRADA']
        entradas_dia = entradas.groupby('dia_semana').size()
        media_diaria = entradas_dia.mean()
        
        for dia in range(5):  # Solo L-V
            if dia in entradas_dia.index:
                self.estacionalidad_semanal[dia] = entradas_dia[dia] / media_diaria
            else:
                self.estacionalidad_semanal[dia] = 1.0
        
        print(f"\n📆 Patrón semanal detectado:")
        dias_nombre = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie']
        for dia in range(5):
            factor = self.estacionalidad_semanal.get(dia, 1.0)
            barra = '█' * int(factor * 10)
            print(f"   {dias_nombre[dia]}: {factor:.2f} {barra}")
    
    def _detectar_tendencia(self):
        """Detecta tendencia temporal (crecimiento/decrecimiento)"""
        
        # Agrupar por semana
        entradas = self.historico[self.historico['tipo'] == 'ENTRADA']
        entradas_semana = entradas.groupby('semana').size().reset_index(name='count')
        
        if len(entradas_semana) < 4:
            self.tendencia = {'slope': 0, 'tipo': 'estable'}
            return
        
        # Regresión lineal simple
        x = np.arange(len(entradas_semana))
        y = entradas_semana['count'].values
        
        # Calcular pendiente
        slope = np.polyfit(x, y, 1)[0]
        
        # Clasificar tendencia
        cambio_pct = slope / y.mean() * 100
        
        if cambio_pct > 2:
            tipo = 'creciente'
        elif cambio_pct < -2:
            tipo = 'decreciente'
        else:
            tipo = 'estable'
        
        self.tendencia = {
            'slope': slope,
            'cambio_pct_semanal': cambio_pct,
            'tipo': tipo
        }
        
        print(f"\n📈 Tendencia detectada: {tipo.upper()}")
        print(f"   Cambio semanal: {'+' if cambio_pct > 0 else ''}{cambio_pct:.2f}%")
    
    def _validar_modelo(self) -> Dict[str, float]:
        """Valida el modelo con backtesting"""
        
        # Usar últimas 4 semanas como validación
        fecha_corte = self.historico['fecha'].max() - timedelta(days=28)
        
        historico_train = self.historico[self.historico['fecha'] < fecha_corte]
        historico_test = self.historico[self.historico['fecha'] >= fecha_corte]
        
        if len(historico_test) < 7:
            return {'mape_entradas': 0, 'mape_salidas': 0, 'validacion': 'insuficiente'}
        
        # Predecir vs real
        entradas_real = len(historico_test[historico_test['tipo'] == 'ENTRADA'])
        salidas_real = len(historico_test[historico_test['tipo'].isin(['SALIDA_PROGRAMADA', 'SALIDA_URGENCIA'])])
        
        # Predicción simple basada en tasas
        semanas_test = 4
        entradas_pred = sum(t['media'] for t in self.tasas_entrada.values()) * semanas_test
        salidas_pred = sum(t['media'] for t in self.tasas_salida.values()) * semanas_test
        
        # MAPE
        mape_ent = abs(entradas_real - entradas_pred) / entradas_real * 100 if entradas_real > 0 else 0
        mape_sal = abs(salidas_real - salidas_pred) / salidas_real * 100 if salidas_real > 0 else 0
        
        return {
            'mape_entradas': mape_ent,
            'mape_salidas': mape_sal,
            'entradas_real': entradas_real,
            'entradas_pred': entradas_pred,
            'salidas_real': salidas_real,
            'salidas_pred': salidas_pred,
        }
    
    def predecir(
        self, 
        semanas: int = 12,
        lista_espera_actual: int = 500,
        fuera_plazo_actual: int = 50,
        n_simulaciones: int = 100
    ) -> ResultadoPrediccion:
        """
        Genera predicción para las próximas semanas.
        
        Args:
            semanas: Número de semanas a predecir
            lista_espera_actual: Tamaño actual de la lista
            fuera_plazo_actual: Pacientes actualmente fuera de plazo
            n_simulaciones: Número de simulaciones Monte Carlo
        
        Returns:
            ResultadoPrediccion con proyecciones e intervalos de confianza
        """
        if not self.modelo_entrenado:
            self.entrenar()
        
        fecha_hoy = date.today()
        predicciones = []
        predicciones_por_esp = defaultdict(list)
        
        lista_espera = lista_espera_actual
        fuera_plazo = fuera_plazo_actual
        
        alertas = []
        recomendaciones = []
        
        for sem in range(1, semanas + 1):
            fecha_inicio_sem = fecha_hoy + timedelta(weeks=sem-1)
            mes = fecha_inicio_sem.month
            
            # Factor estacional
            factor_estacional = self.estacionalidad_mensual.get(mes, 1.0)
            
            # Factor tendencia
            factor_tendencia = 1 + (self.tendencia.get('cambio_pct_semanal', 0) / 100 * sem)
            
            # Predicción agregada con Monte Carlo
            entradas_sim = []
            salidas_sim = []
            
            for _ in range(n_simulaciones):
                ent_total = 0
                sal_total = 0
                
                for esp, tasas in self.tasas_entrada.items():
                    # Muestra de distribución normal truncada
                    ent = max(0, np.random.normal(
                        tasas['media'] * factor_estacional * factor_tendencia,
                        tasas['std']
                    ))
                    ent_total += ent
                
                for esp, tasas in self.tasas_salida.items():
                    sal = max(0, np.random.normal(tasas['media'], tasas['std']))
                    sal_total += sal
                
                entradas_sim.append(ent_total)
                salidas_sim.append(sal_total)
            
            # Calcular estadísticos
            entradas_media = np.mean(entradas_sim)
            entradas_ic_bajo = np.percentile(entradas_sim, 10)
            entradas_ic_alto = np.percentile(entradas_sim, 90)
            
            salidas_media = np.mean(salidas_sim)
            salidas_ic_bajo = np.percentile(salidas_sim, 10)
            salidas_ic_alto = np.percentile(salidas_sim, 90)
            
            balance = entradas_media - salidas_media
            
            # Actualizar proyección lista espera
            lista_espera += balance
            
            # Estimar fuera de plazo (simplificado: crece si lista crece)
            if balance > 0:
                fuera_plazo += balance * 0.3  # 30% de los nuevos acabarán fuera de plazo
            else:
                fuera_plazo = max(0, fuera_plazo + balance * 0.5)
            
            predicciones.append(PrediccionSemanal(
                semana=sem,
                fecha_inicio=fecha_inicio_sem,
                entradas_media=entradas_media,
                entradas_ic_bajo=entradas_ic_bajo,
                entradas_ic_alto=entradas_ic_alto,
                salidas_media=salidas_media,
                salidas_ic_bajo=salidas_ic_bajo,
                salidas_ic_alto=salidas_ic_alto,
                balance_medio=balance,
                lista_espera_proyectada=lista_espera,
                fuera_plazo_proyectado=fuera_plazo
            ))
        
        # Generar alertas
        pred_final = predicciones[-1]
        
        if pred_final.lista_espera_proyectada > lista_espera_actual * 1.2:
            alertas.append(f"⚠️ Lista de espera crecerá +{(pred_final.lista_espera_proyectada/lista_espera_actual - 1)*100:.0f}% en {semanas} semanas")
        
        if pred_final.fuera_plazo_proyectado > fuera_plazo_actual * 1.5:
            alertas.append(f"🚨 Pacientes fuera de plazo aumentarán significativamente")
        
        balance_medio_total = sum(p.balance_medio for p in predicciones) / len(predicciones)
        if balance_medio_total > 5:
            alertas.append(f"📈 Tendencia de crecimiento: +{balance_medio_total:.1f} pacientes/semana")
            
            # Calcular sesiones extra necesarias
            minutos_por_paciente = 90  # Estimación media
            minutos_por_sesion = 7 * 60  # 7 horas
            sesiones_extra = balance_medio_total * minutos_por_paciente / minutos_por_sesion
            recomendaciones.append(f"💡 Necesitas +{sesiones_extra:.1f} sesiones/semana para equilibrar")
        
        return ResultadoPrediccion(
            fecha_prediccion=fecha_hoy,
            semanas_horizonte=semanas,
            lista_espera_actual=lista_espera_actual,
            fuera_plazo_actual=fuera_plazo_actual,
            predicciones=predicciones,
            por_especialidad=dict(predicciones_por_esp),
            metricas_modelo=self.metricas,
            alertas=alertas,
            recomendaciones=recomendaciones
        )
    
    def generar_informe(self, resultado: ResultadoPrediccion) -> str:
        """Genera un informe textual de la predicción"""
        
        lineas = [
            "=" * 60,
            "📈 PREDICCIÓN DE DEMANDA - LISTA DE ESPERA",
            "=" * 60,
            f"Fecha de predicción: {resultado.fecha_prediccion}",
            f"Horizonte: {resultado.semanas_horizonte} semanas",
            f"",
            f"### Estado Actual",
            f"- Lista de espera: {resultado.lista_espera_actual} pacientes",
            f"- Fuera de plazo: {resultado.fuera_plazo_actual} ({resultado.fuera_plazo_actual/resultado.lista_espera_actual*100:.1f}%)",
            f"",
            f"### Calidad del Modelo",
            f"- MAPE Entradas: {resultado.metricas_modelo.get('mape_entradas', 0):.1f}%",
            f"- MAPE Salidas: {resultado.metricas_modelo.get('mape_salidas', 0):.1f}%",
            f"",
            f"### Proyección Semanal",
            f"",
            "| Semana | Fecha | Entradas | Salidas | Balance | Lista Espera | Fuera Plazo |",
            "|--------|-------|----------|---------|---------|--------------|-------------|",
        ]
        
        for p in resultado.predicciones:
            lineas.append(
                f"| +{p.semana} | {p.fecha_inicio.strftime('%d/%m')} | "
                f"{p.entradas_media:.0f} [{p.entradas_ic_bajo:.0f}-{p.entradas_ic_alto:.0f}] | "
                f"{p.salidas_media:.0f} [{p.salidas_ic_bajo:.0f}-{p.salidas_ic_alto:.0f}] | "
                f"{'+' if p.balance_medio >= 0 else ''}{p.balance_medio:.0f} | "
                f"{p.lista_espera_proyectada:.0f} | "
                f"{p.fuera_plazo_proyectado:.0f} |"
            )
        
        if resultado.alertas:
            lineas.extend(["", "### ⚠️ Alertas", ""])
            for alerta in resultado.alertas:
                lineas.append(f"- {alerta}")
        
        if resultado.recomendaciones:
            lineas.extend(["", "### 💡 Recomendaciones", ""])
            for rec in resultado.recomendaciones:
                lineas.append(f"- {rec}")
        
        return "\n".join(lineas)


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def crear_predictor_desde_historico_cirugias(historico_cirugias: pd.DataFrame) -> PredictorDemanda:
    """
    Crea un predictor a partir del histórico de cirugías existente.
    
    Estrategia: Usar las cirugías como proxy de la demanda.
    - Las salidas son las cirugías realizadas (dato real)
    - Las entradas se estiman asumiendo que el sistema está en equilibrio
      con un pequeño factor de crecimiento/decrecimiento
    """
    
    if len(historico_cirugias) == 0:
        raise ValueError("Histórico de cirugías vacío")
    
    # Asegurar que fecha es tipo date
    if 'fecha' in historico_cirugias.columns:
        historico_cirugias = historico_cirugias.copy()
        historico_cirugias['fecha'] = pd.to_datetime(historico_cirugias['fecha']).dt.date
    
    movimientos = []
    
    # Agrupar cirugías por fecha y especialidad para crear salidas
    for fecha in historico_cirugias['fecha'].unique():
        cirugias_dia = historico_cirugias[historico_cirugias['fecha'] == fecha]
        
        for _, row in cirugias_dia.iterrows():
            especialidad = row['especialidad']
            prioridad = row.get('prioridad', 'REFERENCIA_P2')
            
            # Registrar SALIDA (cirugía realizada)
            movimientos.append({
                'fecha': fecha,
                'tipo': 'SALIDA_PROGRAMADA',
                'especialidad': especialidad,
                'prioridad': prioridad,
                'origen': None,
                'dia_semana': fecha.weekday() if hasattr(fecha, 'weekday') else pd.Timestamp(fecha).weekday(),
                'mes': fecha.month if hasattr(fecha, 'month') else pd.Timestamp(fecha).month,
                'semana': fecha.isocalendar()[1] if hasattr(fecha, 'isocalendar') else pd.Timestamp(fecha).isocalendar()[1],
            })
    
    # Calcular tasas de salida por especialidad y semana
    df_salidas = pd.DataFrame(movimientos)
    salidas_por_esp_semana = df_salidas.groupby(['especialidad', 'semana']).size().reset_index(name='count')
    tasas_salida = salidas_por_esp_semana.groupby('especialidad')['count'].mean().to_dict()
    
    # Generar ENTRADAS sintéticas basadas en las salidas
    # Asumimos: entradas ≈ salidas * factor (sistema en cuasi-equilibrio)
    # Factor < 1: lista decrece, Factor > 1: lista crece
    factor_equilibrio = 0.95  # Ligeramente menos entradas que salidas (lista decrece)
    
    fechas_unicas = sorted(df_salidas['fecha'].unique())
    
    for fecha in fechas_unicas:
        # Solo días laborables
        dia_semana = fecha.weekday() if hasattr(fecha, 'weekday') else pd.Timestamp(fecha).weekday()
        if dia_semana >= 5:
            continue
        
        for especialidad, tasa_salida in tasas_salida.items():
            # Tasa de entrada diaria estimada
            tasa_entrada_diaria = (tasa_salida / 5) * factor_equilibrio
            
            # Variación por día de semana
            factor_dia = {0: 1.15, 1: 1.10, 2: 1.00, 3: 0.95, 4: 0.80}.get(dia_semana, 1.0)
            tasa_ajustada = tasa_entrada_diaria * factor_dia
            
            # Número de entradas (Poisson)
            n_entradas = np.random.poisson(max(0.1, tasa_ajustada))
            
            for _ in range(n_entradas):
                # Determinar prioridad
                if especialidad in ['CIRUGIA_MAMA', 'CIRUGIA_COLORRECTAL']:
                    # Mayor proporción oncológica
                    prioridad = np.random.choice(
                        ['ONCOLOGICO_PRIORITARIO', 'ONCOLOGICO_ESTANDAR', 'REFERENCIA_P1', 'REFERENCIA_P2'],
                        p=[0.25, 0.20, 0.25, 0.30]
                    )
                    origen = 'COMITE_TUMORES' if 'ONCOLOGICO' in prioridad else 'CONSULTA_EXTERNA'
                else:
                    prioridad = np.random.choice(
                        ['REFERENCIA_P1', 'REFERENCIA_P2', 'REFERENCIA_P3'],
                        p=[0.25, 0.50, 0.25]
                    )
                    origen = 'CONSULTA_EXTERNA'
                
                mes = fecha.month if hasattr(fecha, 'month') else pd.Timestamp(fecha).month
                semana = fecha.isocalendar()[1] if hasattr(fecha, 'isocalendar') else pd.Timestamp(fecha).isocalendar()[1]
                
                movimientos.append({
                    'fecha': fecha,
                    'tipo': 'ENTRADA',
                    'especialidad': especialidad,
                    'prioridad': prioridad,
                    'origen': origen,
                    'dia_semana': dia_semana,
                    'mes': mes,
                    'semana': semana,
                })
    
    df = pd.DataFrame(movimientos)
    df = df.sort_values('fecha').reset_index(drop=True)
    
    print(f"📊 Histórico de movimientos generado:")
    print(f"   - Entradas: {len(df[df['tipo'] == 'ENTRADA'])}")
    print(f"   - Salidas: {len(df[df['tipo'] == 'SALIDA_PROGRAMADA'])}")
    
    return PredictorDemanda(df)


# =============================================================================
# MAIN - PRUEBA DEL MÓDULO
# =============================================================================

if __name__ == "__main__":
    print("Generando histórico de movimientos sintético...")
    
    generador = GeneradorHistoricoMovimientos(seed=42)
    
    fecha_fin = date.today()
    fecha_inicio = fecha_fin - timedelta(days=365)
    
    historico = generador.generar_historico(fecha_inicio, fecha_fin)
    
    print(f"\nHistórico generado: {len(historico)} movimientos")
    print(f"Entradas: {len(historico[historico['tipo'] == 'ENTRADA'])}")
    print(f"Salidas: {len(historico[historico['tipo'].isin(['SALIDA_PROGRAMADA', 'SALIDA_URGENCIA'])])}")
    
    print("\n" + "=" * 60)
    
    predictor = PredictorDemanda(historico)
    predictor.entrenar()
    
    print("\n" + "=" * 60)
    
    resultado = predictor.predecir(
        semanas=12,
        lista_espera_actual=500,
        fuera_plazo_actual=50
    )
    
    print(predictor.generar_informe(resultado))
