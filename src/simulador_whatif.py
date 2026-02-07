"""
Simulador What-If para el Programador Quirúrgico
================================================
Permite simular escenarios hipotéticos y ver su impacto en:
- Lista de espera
- Pacientes fuera de plazo
- Tiempos de espera
- Necesidad de recursos

Modelos utilizados:
1. Modelo de Capacidad (determinista)
2. Modelo de Demanda (estocástico) 
3. Modelo de Colas M/M/c
4. Simulación Monte Carlo
5. Optimización inversa
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from math import factorial, exp
import warnings


# =============================================================================
# DEFINICIÓN DE ESCENARIOS
# =============================================================================

class TipoEscenario(Enum):
    """Tipos de escenarios what-if"""
    AÑADIR_SESIONES = "añadir_sesiones"
    QUITAR_SESIONES = "quitar_sesiones"
    CERRAR_QUIROFANO = "cerrar_quirofano"
    AUMENTAR_DEMANDA = "aumentar_demanda"
    REDUCIR_DEMANDA = "reducir_demanda"
    CAMBIAR_RESERVA_URGENCIAS = "cambiar_reserva_urgencias"
    PERSONALIZADO = "personalizado"


@dataclass
class Escenario:
    """Define un escenario what-if"""
    nombre: str
    tipo: TipoEscenario
    descripcion: str = ""
    
    # Modificadores de sesiones: {especialidad: num_sesiones_extra}
    sesiones_extra: Dict[str, int] = field(default_factory=dict)
    
    # Quirófanos cerrados: [(quirofano_id, fecha_inicio, fecha_fin)]
    quirofanos_cerrados: List[Tuple[int, date, date]] = field(default_factory=list)
    
    # Días específicos cerrados: [fecha1, fecha2, ...]
    dias_cerrados: List[date] = field(default_factory=list)
    
    # Factor de demanda (1.0 = sin cambio, 1.2 = +20%)
    factor_demanda: float = 1.0
    
    # Factor de demanda por especialidad
    factor_demanda_esp: Dict[str, float] = field(default_factory=dict)
    
    # Cambio en reserva de urgencias (puntos porcentuales)
    cambio_reserva_urgencias: float = 0.0
    
    # Duración del escenario en semanas
    semanas_duracion: int = 12


@dataclass 
class ResultadoSimulacion:
    """Resultado de una simulación what-if"""
    escenario: Escenario
    
    # Proyecciones semanales
    semanas: List[int]
    lista_espera: List[float]  # Por semana
    fuera_plazo: List[float]   # Por semana
    
    # Intervalos de confianza
    lista_ic_bajo: List[float]
    lista_ic_alto: List[float]
    fp_ic_bajo: List[float]
    fp_ic_alto: List[float]
    
    # Métricas agregadas
    lista_final_media: float
    lista_final_p10: float
    lista_final_p90: float
    
    fp_final_media: float
    fp_final_p10: float
    fp_final_p90: float
    
    # Probabilidades
    prob_reducir_lista: float
    prob_eliminar_fp: float
    
    # Comparación con baseline
    diferencia_lista: float
    diferencia_fp: float
    
    # Métricas de colas
    tiempo_espera_medio: float
    utilizacion_sistema: float
    
    # Recomendaciones
    recomendaciones: List[str] = field(default_factory=list)


# =============================================================================
# MODELO DE CAPACIDAD
# =============================================================================

class ModeloCapacidad:
    """
    Calcula la capacidad quirúrgica bajo diferentes configuraciones.
    Modelo determinista basado en sesiones programadas.
    """
    
    # Duración de turnos en minutos
    DURACION_MANANA = 420  # 7 horas
    DURACION_TARDE = 300   # 5 horas
    
    # Duración media de cirugía por especialidad (minutos)
    DURACION_MEDIA_CIRUGIA = {
        'CIRUGIA_GENERAL': 60,
        'CIRUGIA_DIGESTIVA': 90,
        'CIRUGIA_HEPATOBILIAR': 180,
        'CIRUGIA_COLORRECTAL': 150,
        'CIRUGIA_MAMA': 75,
        'CIRUGIA_ENDOCRINA': 100,
        'CIRUGIA_BARIATRICA': 140,
        'UROLOGIA': 70,
        'GINECOLOGIA': 90,
        'CIRUGIA_VASCULAR': 100,
        'CIRUGIA_PLASTICA': 60,
    }
    
    # Tiempo entre cirugías (limpieza + preparación)
    TIEMPO_ENTRE_CIRUGIAS = 30
    
    def __init__(self, configuracion_sesiones: Dict, reservas_urgencias: Dict[str, float] = None):
        """
        Args:
            configuracion_sesiones: Dict[quirofano][dia][turno] = especialidad
            reservas_urgencias: Dict[especialidad] = porcentaje_reserva (0-100)
        """
        self.config = configuracion_sesiones
        self.reservas = reservas_urgencias or {}
    
    def calcular_capacidad_semanal(self, escenario: Escenario = None) -> Dict[str, Dict]:
        """
        Calcula la capacidad semanal por especialidad.
        
        Returns:
            Dict[especialidad] = {
                'minutos_brutos': int,
                'minutos_efectivos': int,
                'cirugias_estimadas': float,
                'sesiones': int
            }
        """
        capacidad = defaultdict(lambda: {
            'minutos_brutos': 0,
            'minutos_efectivos': 0,
            'cirugias_estimadas': 0,
            'sesiones': 0
        })
        
        dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
        turnos = ['Mañana', 'Tarde']
        
        for q in self.config:
            for dia in dias_semana:
                for turno in turnos:
                    esp = self.config[q][dia][turno]
                    
                    if esp in ['LIBRE', 'CERRADO']:
                        continue
                    
                    # Verificar si quirófano está cerrado en escenario
                    if escenario and self._quirofano_cerrado(q, dia, escenario):
                        continue
                    
                    duracion = self.DURACION_MANANA if turno == 'Mañana' else self.DURACION_TARDE
                    capacidad[esp]['minutos_brutos'] += duracion
                    capacidad[esp]['sesiones'] += 1
        
        # Aplicar sesiones extra del escenario
        if escenario:
            for esp, num_extra in escenario.sesiones_extra.items():
                minutos_extra = num_extra * self.DURACION_MANANA  # Asumir sesiones de mañana
                capacidad[esp]['minutos_brutos'] += minutos_extra
                capacidad[esp]['sesiones'] += num_extra
        
        # Calcular capacidad efectiva y cirugías estimadas
        for esp in capacidad:
            brutos = capacidad[esp]['minutos_brutos']
            
            # Aplicar reserva de urgencias
            reserva_pct = self.reservas.get(esp, 15) / 100
            
            # Modificar reserva si está en el escenario
            if escenario and escenario.cambio_reserva_urgencias != 0:
                reserva_pct = max(0, min(0.5, reserva_pct + escenario.cambio_reserva_urgencias / 100))
            
            efectivos = brutos * (1 - reserva_pct)
            capacidad[esp]['minutos_efectivos'] = efectivos
            
            # Estimar número de cirugías
            dur_media = self.DURACION_MEDIA_CIRUGIA.get(esp, 90)
            cirugias = efectivos / (dur_media + self.TIEMPO_ENTRE_CIRUGIAS)
            capacidad[esp]['cirugias_estimadas'] = cirugias
        
        return dict(capacidad)
    
    def _quirofano_cerrado(self, quirofano: int, dia: str, escenario: Escenario) -> bool:
        """Verifica si un quirófano está cerrado en el escenario"""
        # Por ahora, simplificado - se puede extender para fechas específicas
        for q, _, _ in escenario.quirofanos_cerrados:
            if q == quirofano:
                return True
        return False
    
    def calcular_diferencia(self, escenario: Escenario) -> Dict[str, Dict]:
        """Calcula la diferencia de capacidad entre baseline y escenario"""
        cap_base = self.calcular_capacidad_semanal()
        cap_escenario = self.calcular_capacidad_semanal(escenario)
        
        diferencia = {}
        todas_esp = set(cap_base.keys()) | set(cap_escenario.keys())
        
        for esp in todas_esp:
            base = cap_base.get(esp, {'cirugias_estimadas': 0, 'sesiones': 0})
            nuevo = cap_escenario.get(esp, {'cirugias_estimadas': 0, 'sesiones': 0})
            
            diferencia[esp] = {
                'sesiones_base': base['sesiones'],
                'sesiones_nuevo': nuevo['sesiones'],
                'sesiones_diff': nuevo['sesiones'] - base['sesiones'],
                'cirugias_base': base['cirugias_estimadas'],
                'cirugias_nuevo': nuevo['cirugias_estimadas'],
                'cirugias_diff': nuevo['cirugias_estimadas'] - base['cirugias_estimadas']
            }
        
        return diferencia


# =============================================================================
# MODELO DE COLAS M/M/c
# =============================================================================

class ModeloColas:
    """
    Modelo de teoría de colas para analizar tiempos de espera.
    Implementa M/M/c (llegadas Poisson, servicio exponencial, c servidores)
    """
    
    @staticmethod
    def calcular_metricas(lambda_llegadas: float, mu_servicio: float, 
                          c_servidores: int) -> Dict[str, float]:
        """
        Calcula métricas del sistema de colas M/M/c.
        
        Args:
            lambda_llegadas: Tasa de llegadas (pacientes/semana)
            mu_servicio: Tasa de servicio por servidor (cirugías/semana)
            c_servidores: Número de servidores (sesiones equivalentes)
        
        Returns:
            Dict con métricas del sistema
        """
        if c_servidores <= 0 or mu_servicio <= 0:
            return {
                'estado': 'INVALIDO',
                'utilizacion': 1.0,
                'cola_media': float('inf'),
                'espera_media_dias': float('inf'),
                'prob_espera': 1.0
            }
        
        rho = lambda_llegadas / (c_servidores * mu_servicio)
        
        # Sistema inestable si rho >= 1
        if rho >= 1:
            return {
                'estado': 'INESTABLE',
                'utilizacion': rho,
                'cola_media': float('inf'),
                'espera_media_dias': float('inf'),
                'prob_espera': 1.0,
                'mensaje': 'La demanda supera la capacidad. Lista crecerá indefinidamente.'
            }
        
        # Calcular P0 (probabilidad de sistema vacío)
        try:
            suma = sum((c_servidores * rho)**n / factorial(n) for n in range(c_servidores))
            ultimo_termino = (c_servidores * rho)**c_servidores / (factorial(c_servidores) * (1 - rho))
            p0 = 1.0 / (suma + ultimo_termino)
        except:
            p0 = 0.01
        
        # Probabilidad de espera (Erlang C)
        try:
            prob_espera = ((c_servidores * rho)**c_servidores / 
                          (factorial(c_servidores) * (1 - rho))) * p0
        except:
            prob_espera = 0.5
        
        # Longitud media de cola (Lq)
        Lq = prob_espera * rho / (1 - rho)
        
        # Tiempo medio en cola (Wq) en semanas
        Wq = Lq / lambda_llegadas if lambda_llegadas > 0 else 0
        
        # Convertir a días
        espera_dias = Wq * 7
        
        return {
            'estado': 'ESTABLE',
            'utilizacion': rho,
            'cola_media': Lq,
            'espera_media_dias': espera_dias,
            'prob_espera': min(1.0, prob_espera),
            'p0': p0
        }
    
    @staticmethod
    def prob_superar_tiempo(tiempo_dias: float, mu_servicio: float, 
                            prob_espera: float) -> float:
        """
        Calcula la probabilidad de superar un tiempo de espera dado.
        
        Args:
            tiempo_dias: Tiempo límite en días
            mu_servicio: Tasa de servicio
            prob_espera: Probabilidad de tener que esperar
        
        Returns:
            Probabilidad de superar el tiempo
        """
        if tiempo_dias <= 0:
            return prob_espera
        
        tiempo_semanas = tiempo_dias / 7
        return prob_espera * exp(-mu_servicio * tiempo_semanas * (1 - prob_espera))


# =============================================================================
# SIMULADOR MONTE CARLO
# =============================================================================

class SimuladorMonteCarlo:
    """
    Simulador Monte Carlo para proyectar la evolución de la lista de espera
    bajo diferentes escenarios con incertidumbre.
    """
    
    def __init__(self, 
                 lista_actual: int,
                 fuera_plazo_actual: int,
                 tasas_entrada: Dict[str, float],
                 tasas_salida: Dict[str, float],
                 seed: int = 42):
        """
        Args:
            lista_actual: Número actual de pacientes en lista
            fuera_plazo_actual: Pacientes actualmente fuera de plazo
            tasas_entrada: Entradas semanales por especialidad
            tasas_salida: Salidas semanales por especialidad (capacidad)
            seed: Semilla para reproducibilidad
        """
        self.lista_actual = lista_actual
        self.fp_actual = fuera_plazo_actual
        self.tasas_entrada = tasas_entrada
        self.tasas_salida = tasas_salida
        np.random.seed(seed)
    
    def simular(self, escenario: Escenario, n_simulaciones: int = 500) -> ResultadoSimulacion:
        """
        Ejecuta simulación Monte Carlo del escenario.
        
        Args:
            escenario: Escenario a simular
            n_simulaciones: Número de simulaciones
        
        Returns:
            ResultadoSimulacion con proyecciones e intervalos
        """
        semanas = escenario.semanas_duracion
        
        # Ajustar tasas según escenario
        tasas_entrada_adj = self._ajustar_tasas_entrada(escenario)
        tasas_salida_adj = self._ajustar_tasas_salida(escenario)
        
        # Almacenar resultados de cada simulación
        todas_listas = []
        todos_fp = []
        
        for _ in range(n_simulaciones):
            lista, fp = self._simular_una_vez(semanas, tasas_entrada_adj, tasas_salida_adj)
            todas_listas.append(lista)
            todos_fp.append(fp)
        
        # Convertir a arrays para cálculos
        todas_listas = np.array(todas_listas)  # Shape: (n_sim, semanas+1)
        todos_fp = np.array(todos_fp)
        
        # Calcular estadísticos por semana
        lista_media = np.mean(todas_listas, axis=0)
        lista_p10 = np.percentile(todas_listas, 10, axis=0)
        lista_p90 = np.percentile(todas_listas, 90, axis=0)
        
        fp_media = np.mean(todos_fp, axis=0)
        fp_p10 = np.percentile(todos_fp, 10, axis=0)
        fp_p90 = np.percentile(todos_fp, 90, axis=0)
        
        # Métricas finales
        lista_final = todas_listas[:, -1]
        fp_final = todos_fp[:, -1]
        
        # Probabilidades
        prob_reducir = np.mean(lista_final < self.lista_actual)
        prob_eliminar_fp = np.mean(fp_final <= 5)  # Consideramos "eliminado" si <= 5
        
        # Calcular baseline (sin escenario)
        escenario_base = Escenario(nombre="baseline", tipo=TipoEscenario.PERSONALIZADO)
        baseline_lista, baseline_fp = self._simular_una_vez(
            semanas, self.tasas_entrada, self.tasas_salida
        )
        
        # Métricas de colas
        lambda_total = sum(tasas_entrada_adj.values())
        mu_total = sum(tasas_salida_adj.values())
        c_equiv = max(1, int(mu_total / 5))  # Servidores equivalentes
        
        metricas_colas = ModeloColas.calcular_metricas(lambda_total, mu_total / c_equiv, c_equiv)
        
        # Generar recomendaciones
        recomendaciones = self._generar_recomendaciones(
            lista_media[-1], fp_media[-1], 
            baseline_lista[-1], baseline_fp[-1],
            metricas_colas
        )
        
        return ResultadoSimulacion(
            escenario=escenario,
            semanas=list(range(semanas + 1)),
            lista_espera=lista_media.tolist(),
            fuera_plazo=fp_media.tolist(),
            lista_ic_bajo=lista_p10.tolist(),
            lista_ic_alto=lista_p90.tolist(),
            fp_ic_bajo=fp_p10.tolist(),
            fp_ic_alto=fp_p90.tolist(),
            lista_final_media=float(np.mean(lista_final)),
            lista_final_p10=float(np.percentile(lista_final, 10)),
            lista_final_p90=float(np.percentile(lista_final, 90)),
            fp_final_media=float(np.mean(fp_final)),
            fp_final_p10=float(np.percentile(fp_final, 10)),
            fp_final_p90=float(np.percentile(fp_final, 90)),
            prob_reducir_lista=float(prob_reducir),
            prob_eliminar_fp=float(prob_eliminar_fp),
            diferencia_lista=float(lista_media[-1] - baseline_lista[-1]),
            diferencia_fp=float(fp_media[-1] - baseline_fp[-1]),
            tiempo_espera_medio=metricas_colas.get('espera_media_dias', 0),
            utilizacion_sistema=metricas_colas.get('utilizacion', 0),
            recomendaciones=recomendaciones
        )
    
    def _ajustar_tasas_entrada(self, escenario: Escenario) -> Dict[str, float]:
        """Ajusta tasas de entrada según el escenario"""
        ajustadas = {}
        for esp, tasa in self.tasas_entrada.items():
            factor = escenario.factor_demanda
            factor_esp = escenario.factor_demanda_esp.get(esp, 1.0)
            ajustadas[esp] = tasa * factor * factor_esp
        return ajustadas
    
    def _ajustar_tasas_salida(self, escenario: Escenario) -> Dict[str, float]:
        """Ajusta tasas de salida según el escenario (sesiones extra, etc.)"""
        ajustadas = dict(self.tasas_salida)
        
        # Añadir capacidad por sesiones extra
        for esp, num_sesiones in escenario.sesiones_extra.items():
            cirugias_por_sesion = 4  # Estimación conservadora
            if esp in ajustadas:
                ajustadas[esp] += num_sesiones * cirugias_por_sesion
            else:
                ajustadas[esp] = num_sesiones * cirugias_por_sesion
        
        # Reducir capacidad por quirófanos cerrados
        if escenario.quirofanos_cerrados:
            factor_reduccion = 1 - (len(escenario.quirofanos_cerrados) * 0.125)  # ~12.5% por quirófano
            for esp in ajustadas:
                ajustadas[esp] *= max(0.5, factor_reduccion)
        
        return ajustadas
    
    def _simular_una_vez(self, semanas: int, 
                         tasas_entrada: Dict[str, float],
                         tasas_salida: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """Ejecuta una simulación completa"""
        lista = [self.lista_actual]
        fp = [self.fp_actual]
        
        total_entrada = sum(tasas_entrada.values())
        total_salida = sum(tasas_salida.values())
        
        for sem in range(semanas):
            # Simular entradas (Poisson)
            entradas = np.random.poisson(total_entrada)
            
            # Simular salidas con variabilidad (cancelaciones, etc.)
            factor_variabilidad = np.random.uniform(0.85, 1.05)
            salidas = int(total_salida * factor_variabilidad)
            
            # Actualizar lista
            nueva_lista = max(0, lista[-1] + entradas - salidas)
            lista.append(nueva_lista)
            
            # Estimar fuera de plazo
            # Simplificación: los FP crecen si lista crece, decrecen si lista decrece
            if entradas > salidas:
                nuevos_fp = fp[-1] + (entradas - salidas) * 0.25  # 25% de nuevos serán FP
            else:
                reduccion = min(fp[-1], (salidas - entradas) * 0.4)  # Priorizamos FP
                nuevos_fp = fp[-1] - reduccion
            
            # Además, cada semana algunos pasan a FP por tiempo
            conversion_fp = lista[-1] * 0.02  # 2% pasan a FP cada semana
            nuevos_fp = max(0, nuevos_fp + conversion_fp)
            nuevos_fp = min(nuevos_fp, nueva_lista)  # No puede haber más FP que lista
            
            fp.append(nuevos_fp)
        
        return lista, fp
    
    def _generar_recomendaciones(self, lista_final: float, fp_final: float,
                                  baseline_lista: float, baseline_fp: float,
                                  metricas_colas: Dict) -> List[str]:
        """Genera recomendaciones basadas en los resultados"""
        recs = []
        
        diff_lista = lista_final - baseline_lista
        diff_fp = fp_final - baseline_fp
        
        if diff_lista < -20:
            recs.append(f"✅ Este escenario reduce la lista en ~{abs(diff_lista):.0f} pacientes")
        elif diff_lista > 20:
            recs.append(f"⚠️ Este escenario aumenta la lista en ~{diff_lista:.0f} pacientes")
        
        if diff_fp < -10:
            recs.append(f"✅ Reduce pacientes fuera de plazo en ~{abs(diff_fp):.0f}")
        elif diff_fp > 10:
            recs.append(f"🚨 Aumenta pacientes fuera de plazo en ~{diff_fp:.0f}")
        
        utilizacion = metricas_colas.get('utilizacion', 0)
        if utilizacion > 0.95:
            recs.append("🔴 Sistema saturado (>95% utilización). Añadir capacidad urgente.")
        elif utilizacion > 0.85:
            recs.append("🟡 Sistema cerca de saturación (>85%). Monitorizar de cerca.")
        elif utilizacion < 0.7:
            recs.append("🟢 Sistema con margen de capacidad. Posible optimizar recursos.")
        
        return recs


# =============================================================================
# OPTIMIZACIÓN INVERSA
# =============================================================================

class OptimizadorInverso:
    """
    Encuentra la configuración mínima necesaria para alcanzar un objetivo.
    Por ejemplo: "¿Cuántas sesiones extra necesito para eliminar fuera de plazo?"
    """
    
    def __init__(self, simulador: SimuladorMonteCarlo, modelo_capacidad: ModeloCapacidad):
        self.simulador = simulador
        self.modelo_capacidad = modelo_capacidad
    
    def encontrar_sesiones_minimas(self, 
                                    especialidad: str,
                                    objetivo_fp: int = 0,
                                    semanas: int = 12,
                                    confianza: float = 0.8) -> Dict[str, Any]:
        """
        Encuentra el número mínimo de sesiones extra para alcanzar objetivo.
        
        Args:
            especialidad: Especialidad a la que añadir sesiones
            objetivo_fp: Objetivo de fuera de plazo (default 0)
            semanas: Horizonte temporal
            confianza: Nivel de confianza requerido (0.8 = 80%)
        
        Returns:
            Dict con sesiones necesarias y proyección
        """
        sesiones_min = 0
        sesiones_max = 15
        mejor_resultado = None
        
        while sesiones_max - sesiones_min > 1:
            sesiones_test = (sesiones_min + sesiones_max) // 2
            
            escenario = Escenario(
                nombre=f"Test +{sesiones_test} sesiones",
                tipo=TipoEscenario.AÑADIR_SESIONES,
                sesiones_extra={especialidad: sesiones_test},
                semanas_duracion=semanas
            )
            
            resultado = self.simulador.simular(escenario, n_simulaciones=200)
            
            if resultado.prob_eliminar_fp >= confianza:
                sesiones_max = sesiones_test
                mejor_resultado = resultado
            else:
                sesiones_min = sesiones_test
        
        # Verificar el valor final
        escenario_final = Escenario(
            nombre=f"+{sesiones_max} sesiones de {especialidad}",
            tipo=TipoEscenario.AÑADIR_SESIONES,
            sesiones_extra={especialidad: sesiones_max},
            semanas_duracion=semanas
        )
        resultado_final = self.simulador.simular(escenario_final, n_simulaciones=300)
        
        return {
            'sesiones_necesarias': sesiones_max,
            'especialidad': especialidad,
            'semanas': semanas,
            'prob_exito': resultado_final.prob_eliminar_fp,
            'lista_final_esperada': resultado_final.lista_final_media,
            'fp_final_esperado': resultado_final.fp_final_media,
            'resultado_simulacion': resultado_final
        }
    
    def encontrar_equilibrio(self, semanas_max: int = 24) -> Dict[str, Any]:
        """
        Encuentra en cuántas semanas el sistema alcanza equilibrio
        (lista estable) con la configuración actual.
        """
        escenario_base = Escenario(
            nombre="Baseline",
            tipo=TipoEscenario.PERSONALIZADO,
            semanas_duracion=semanas_max
        )
        
        resultado = self.simulador.simular(escenario_base, n_simulaciones=300)
        
        # Buscar semana donde la lista se estabiliza
        lista = resultado.lista_espera
        semana_equilibrio = None
        
        for i in range(3, len(lista)):
            # Consideramos equilibrio si variación < 2% en 3 semanas
            variacion = abs(lista[i] - lista[i-3]) / max(1, lista[i-3])
            if variacion < 0.02:
                semana_equilibrio = i - 3
                break
        
        tendencia = 'estable'
        if lista[-1] > lista[0] * 1.1:
            tendencia = 'creciente'
        elif lista[-1] < lista[0] * 0.9:
            tendencia = 'decreciente'
        
        return {
            'semana_equilibrio': semana_equilibrio,
            'lista_equilibrio': lista[semana_equilibrio] if semana_equilibrio else lista[-1],
            'tendencia': tendencia,
            'resultado_simulacion': resultado
        }


# =============================================================================
# SIMULADOR PRINCIPAL
# =============================================================================

class SimuladorWhatIf:
    """
    Clase principal que integra todos los modelos de simulación.
    """
    
    def __init__(self, 
                 configuracion_sesiones: Dict,
                 lista_espera_actual: int,
                 fuera_plazo_actual: int,
                 tasas_entrada: Dict[str, float],
                 reservas_urgencias: Dict[str, float] = None):
        """
        Args:
            configuracion_sesiones: Config actual de sesiones
            lista_espera_actual: Tamaño actual de lista
            fuera_plazo_actual: Pacientes fuera de plazo
            tasas_entrada: Entradas semanales por especialidad
            reservas_urgencias: % reserva por especialidad
        """
        self.config_sesiones = configuracion_sesiones
        self.lista_actual = lista_espera_actual
        self.fp_actual = fuera_plazo_actual
        self.tasas_entrada = tasas_entrada
        self.reservas = reservas_urgencias or {}
        
        # Inicializar modelos
        self.modelo_capacidad = ModeloCapacidad(configuracion_sesiones, reservas_urgencias)
        
        # Calcular tasas de salida desde capacidad
        capacidad = self.modelo_capacidad.calcular_capacidad_semanal()
        self.tasas_salida = {esp: datos['cirugias_estimadas'] 
                            for esp, datos in capacidad.items()}
        
        self.simulador_mc = SimuladorMonteCarlo(
            lista_espera_actual, fuera_plazo_actual,
            tasas_entrada, self.tasas_salida
        )
        
        self.optimizador = OptimizadorInverso(self.simulador_mc, self.modelo_capacidad)
    
    def simular_escenario(self, escenario: Escenario, 
                          n_simulaciones: int = 500) -> ResultadoSimulacion:
        """Simula un escenario y devuelve resultados"""
        return self.simulador_mc.simular(escenario, n_simulaciones)
    
    def comparar_escenarios(self, escenarios: List[Escenario],
                            n_simulaciones: int = 300) -> pd.DataFrame:
        """Compara múltiples escenarios"""
        resultados = []
        
        for esc in escenarios:
            res = self.simular_escenario(esc, n_simulaciones)
            resultados.append({
                'Escenario': esc.nombre,
                'Lista Final': f"{res.lista_final_media:.0f}",
                'IC 80%': f"[{res.lista_final_p10:.0f} - {res.lista_final_p90:.0f}]",
                'Fuera Plazo': f"{res.fp_final_media:.0f}",
                'Prob. Reducir Lista': f"{res.prob_reducir_lista:.0%}",
                'Prob. Eliminar FP': f"{res.prob_eliminar_fp:.0%}",
                'Δ Lista': f"{res.diferencia_lista:+.0f}",
                'Δ FP': f"{res.diferencia_fp:+.0f}"
            })
        
        return pd.DataFrame(resultados)
    
    def calcular_sesiones_necesarias(self, especialidad: str,
                                      objetivo_fp: int = 0,
                                      semanas: int = 12) -> Dict:
        """Calcula sesiones necesarias para objetivo"""
        return self.optimizador.encontrar_sesiones_minimas(
            especialidad, objetivo_fp, semanas
        )
    
    def analizar_capacidad(self, escenario: Escenario = None) -> Dict:
        """Analiza la capacidad actual vs escenario"""
        if escenario:
            return self.modelo_capacidad.calcular_diferencia(escenario)
        return self.modelo_capacidad.calcular_capacidad_semanal()
    
    def generar_informe(self, resultado: ResultadoSimulacion) -> str:
        """Genera informe textual del resultado"""
        esc = resultado.escenario
        
        lineas = [
            "=" * 60,
            f"📊 SIMULACIÓN WHAT-IF: {esc.nombre}",
            "=" * 60,
            f"",
            f"**Tipo:** {esc.tipo.value}",
            f"**Horizonte:** {esc.semanas_duracion} semanas",
            f"",
            "### Configuración del Escenario",
        ]
        
        if esc.sesiones_extra:
            lineas.append("**Sesiones extra:**")
            for esp, num in esc.sesiones_extra.items():
                lineas.append(f"  - {esp}: +{num} sesiones/semana")
        
        if esc.quirofanos_cerrados:
            lineas.append(f"**Quirófanos cerrados:** {len(esc.quirofanos_cerrados)}")
        
        if esc.factor_demanda != 1.0:
            lineas.append(f"**Factor demanda:** {esc.factor_demanda:.0%}")
        
        lineas.extend([
            "",
            "### Proyección Lista de Espera",
            "",
            "| Semana | Lista Espera | IC 80% | Fuera Plazo |",
            "|--------|--------------|--------|-------------|",
        ])
        
        for i in [0, 4, 8, min(12, esc.semanas_duracion)]:
            if i < len(resultado.lista_espera):
                lineas.append(
                    f"| {i} | {resultado.lista_espera[i]:.0f} | "
                    f"[{resultado.lista_ic_bajo[i]:.0f}-{resultado.lista_ic_alto[i]:.0f}] | "
                    f"{resultado.fuera_plazo[i]:.0f} |"
                )
        
        lineas.extend([
            "",
            "### Métricas Finales",
            "",
            f"| Métrica | Valor |",
            f"|---------|-------|",
            f"| Lista final (media) | {resultado.lista_final_media:.0f} |",
            f"| Lista final IC 80% | [{resultado.lista_final_p10:.0f} - {resultado.lista_final_p90:.0f}] |",
            f"| Fuera plazo final | {resultado.fp_final_media:.0f} |",
            f"| Δ vs baseline | {resultado.diferencia_lista:+.0f} pacientes |",
            f"| Prob. reducir lista | {resultado.prob_reducir_lista:.0%} |",
            f"| Prob. eliminar FP | {resultado.prob_eliminar_fp:.0%} |",
            f"| Utilización sistema | {resultado.utilizacion_sistema:.0%} |",
        ])
        
        if resultado.recomendaciones:
            lineas.extend(["", "### 💡 Recomendaciones", ""])
            for rec in resultado.recomendaciones:
                lineas.append(f"- {rec}")
        
        return "\n".join(lineas)


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def crear_escenario_rapido(tipo: str, **kwargs) -> Escenario:
    """
    Crea escenarios comunes de forma rápida.
    
    Ejemplos:
        crear_escenario_rapido('añadir', especialidad='UROLOGIA', sesiones=2)
        crear_escenario_rapido('cerrar_quirofano', quirofano=3, semanas=2)
        crear_escenario_rapido('aumento_demanda', porcentaje=20)
    """
    if tipo == 'añadir':
        esp = kwargs.get('especialidad', 'CIRUGIA_GENERAL')
        num = kwargs.get('sesiones', 1)
        return Escenario(
            nombre=f"+{num} sesiones {esp}",
            tipo=TipoEscenario.AÑADIR_SESIONES,
            sesiones_extra={esp: num},
            semanas_duracion=kwargs.get('semanas', 12)
        )
    
    elif tipo == 'quitar':
        esp = kwargs.get('especialidad', 'CIRUGIA_GENERAL')
        num = kwargs.get('sesiones', 1)
        return Escenario(
            nombre=f"-{num} sesiones {esp}",
            tipo=TipoEscenario.QUITAR_SESIONES,
            sesiones_extra={esp: -num},
            semanas_duracion=kwargs.get('semanas', 12)
        )
    
    elif tipo == 'cerrar_quirofano':
        q = kwargs.get('quirofano', 1)
        return Escenario(
            nombre=f"Cerrar Q{q}",
            tipo=TipoEscenario.CERRAR_QUIROFANO,
            quirofanos_cerrados=[(q, date.today(), date.today() + timedelta(weeks=kwargs.get('semanas', 2)))],
            semanas_duracion=kwargs.get('semanas', 12)
        )
    
    elif tipo == 'aumento_demanda':
        pct = kwargs.get('porcentaje', 10)
        return Escenario(
            nombre=f"+{pct}% demanda",
            tipo=TipoEscenario.AUMENTAR_DEMANDA,
            factor_demanda=1 + pct/100,
            semanas_duracion=kwargs.get('semanas', 12)
        )
    
    elif tipo == 'reduccion_demanda':
        pct = kwargs.get('porcentaje', 10)
        return Escenario(
            nombre=f"-{pct}% demanda",
            tipo=TipoEscenario.REDUCIR_DEMANDA,
            factor_demanda=1 - pct/100,
            semanas_duracion=kwargs.get('semanas', 12)
        )
    
    else:
        return Escenario(
            nombre="Escenario personalizado",
            tipo=TipoEscenario.PERSONALIZADO,
            semanas_duracion=kwargs.get('semanas', 12)
        )


# =============================================================================
# MAIN - PRUEBA DEL MÓDULO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA DEL SIMULADOR WHAT-IF")
    print("=" * 60)
    
    # Configuración de ejemplo
    config_sesiones = {
        1: {d: {'Mañana': 'CIRUGIA_DIGESTIVA', 'Tarde': 'CIRUGIA_DIGESTIVA'} 
            for d in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']},
        2: {d: {'Mañana': 'CIRUGIA_GENERAL', 'Tarde': 'CIRUGIA_GENERAL'} 
            for d in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']},
        3: {d: {'Mañana': 'UROLOGIA', 'Tarde': 'UROLOGIA'} 
            for d in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']},
        4: {d: {'Mañana': 'GINECOLOGIA', 'Tarde': 'LIBRE'} 
            for d in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']},
    }
    
    tasas_entrada = {
        'CIRUGIA_DIGESTIVA': 12,
        'CIRUGIA_GENERAL': 15,
        'UROLOGIA': 10,
        'GINECOLOGIA': 8,
    }
    
    reservas = {
        'CIRUGIA_DIGESTIVA': 20,
        'CIRUGIA_GENERAL': 25,
        'UROLOGIA': 15,
        'GINECOLOGIA': 10,
    }
    
    # Crear simulador
    simulador = SimuladorWhatIf(
        configuracion_sesiones=config_sesiones,
        lista_espera_actual=500,
        fuera_plazo_actual=50,
        tasas_entrada=tasas_entrada,
        reservas_urgencias=reservas
    )
    
    # Probar escenarios
    print("\n--- Escenario 1: +2 sesiones Digestivo ---")
    esc1 = crear_escenario_rapido('añadir', especialidad='CIRUGIA_DIGESTIVA', sesiones=2)
    res1 = simulador.simular_escenario(esc1)
    print(simulador.generar_informe(res1))
    
    print("\n--- Escenario 2: Cerrar Q3 ---")
    esc2 = crear_escenario_rapido('cerrar_quirofano', quirofano=3, semanas=4)
    res2 = simulador.simular_escenario(esc2)
    print(simulador.generar_informe(res2))
    
    print("\n--- Escenario 3: +15% demanda ---")
    esc3 = crear_escenario_rapido('aumento_demanda', porcentaje=15)
    res3 = simulador.simular_escenario(esc3)
    print(simulador.generar_informe(res3))
    
    print("\n--- Comparativa ---")
    df = simulador.comparar_escenarios([esc1, esc2, esc3])
    print(df.to_string(index=False))
