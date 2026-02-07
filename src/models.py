"""
Modelos de Datos para el Programador QuirÃºrgico
================================================
DefiniciÃ³n de estructuras de datos para pacientes, cirugÃ­as, 
cirujanos, y programaciÃ³n.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import uuid
import numpy as np

from config import (
    Especialidad, PrioridadCatSalut, TipoIntervencion,
    TIEMPOS_MAXIMOS_ESPERA, CATALOGO_INTERVENCIONES
)


# =============================================================================
# CLASES ASA Y COMORBILIDADES
# =============================================================================

class ClaseASA(Enum):
    """ClasificaciÃ³n ASA de riesgo anestÃ©sico"""
    ASA_I = 1    # Paciente sano
    ASA_II = 2   # Enfermedad sistÃ©mica leve
    ASA_III = 3  # Enfermedad sistÃ©mica grave
    ASA_IV = 4   # Enfermedad sistÃ©mica grave con amenaza vital
    ASA_V = 5    # Moribundo


class Comorbilidad(Enum):
    """Comorbilidades relevantes para programaciÃ³n quirÃºrgica"""
    DIABETES = auto()
    HIPERTENSION = auto()
    CARDIOPATIA = auto()
    EPOC = auto()
    INSUFICIENCIA_RENAL = auto()
    HEPATOPATIA = auto()
    OBESIDAD_MORBIDA = auto()
    ANTICOAGULACION = auto()
    INMUNOSUPRESION = auto()
    ALERGIA_LATEX = auto()


# =============================================================================
# MODELO DE PACIENTE
# =============================================================================

@dataclass
class Paciente:
    """InformaciÃ³n del paciente"""
    id: str
    nombre: str
    fecha_nacimiento: date
    sexo: str  # 'M' o 'F'
    numero_historia: str
    
    # ClasificaciÃ³n clÃ­nica
    clase_asa: ClaseASA = ClaseASA.ASA_I
    comorbilidades: List[Comorbilidad] = field(default_factory=list)
    
    # Datos de contacto (para notificaciones)
    telefono: str = ""
    email: str = ""
    
    # InformaciÃ³n adicional
    notas_clinicas: str = ""
    
    @property
    def edad(self) -> int:
        """Calcula la edad actual del paciente"""
        today = date.today()
        return today.year - self.fecha_nacimiento.year - (
            (today.month, today.day) < (self.fecha_nacimiento.month, self.fecha_nacimiento.day)
        )
    
    @property
    def es_pediatrico(self) -> bool:
        return self.edad < 18
    
    @property
    def es_geriatrico(self) -> bool:
        return self.edad >= 75
    
    def factor_riesgo_edad(self) -> float:
        """Factor de riesgo basado en edad (0-1)"""
        if self.edad < 18:
            return 0.2
        elif self.edad < 45:
            return 0.1
        elif self.edad < 65:
            return 0.3
        elif self.edad < 75:
            return 0.5
        else:
            return 0.7


# =============================================================================
# MODELO DE CIRUJANO
# =============================================================================

@dataclass
class Cirujano:
    """InformaciÃ³n del cirujano"""
    id: str
    nombre: str
    especialidad_principal: Especialidad
    especialidades_secundarias: List[Especialidad] = field(default_factory=list)
    
    # Capacidades
    intervenciones_habilitadas: List[str] = field(default_factory=list)  # CÃ³digos de intervenciÃ³n
    puede_operar_urgencias: bool = True
    nivel_experiencia: int = 3  # 1-5
    
    # Disponibilidad
    dias_disponibles: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # L-V
    quirofanos_preferidos: List[int] = field(default_factory=list)
    
    # Restricciones aprendidas (se llenan automÃ¡ticamente)
    restricciones_aprendidas: Dict[str, Any] = field(default_factory=dict)
    
    def puede_realizar(self, codigo_intervencion: str) -> bool:
        """Verifica si el cirujano puede realizar una intervenciÃ³n"""
        if not self.intervenciones_habilitadas:
            # Si no hay lista especÃ­fica, puede hacer todas de su especialidad
            if codigo_intervencion in CATALOGO_INTERVENCIONES:
                interv = CATALOGO_INTERVENCIONES[codigo_intervencion]
                return (interv.especialidad == self.especialidad_principal or
                       interv.especialidad in self.especialidades_secundarias)
        return codigo_intervencion in self.intervenciones_habilitadas


# =============================================================================
# MODELO DE INTERVENCIÃ“N PROGRAMADA
# =============================================================================

@dataclass
class SolicitudCirugia:
    """
    Solicitud de intervenciÃ³n quirÃºrgica pendiente de programar.
    Representa un paciente en lista de espera.
    """
    id: str
    paciente: Paciente
    tipo_intervencion: TipoIntervencion
    
    # Fechas clave
    fecha_indicacion: date  # Fecha de inclusiÃ³n en lista de espera
    fecha_limite: Optional[date] = None  # Calculada segÃºn prioridad
    
    # PriorizaciÃ³n
    prioridad: PrioridadCatSalut = PrioridadCatSalut.REFERENCIA_P2
    
    # AsignaciÃ³n preferida
    cirujano_solicitante: Optional[Cirujano] = None
    cirujano_asignado: Optional[Cirujano] = None
    
    # InformaciÃ³n clÃ­nica adicional
    diagnostico_principal: str = ""
    lateralidad: Optional[str] = None  # 'izq', 'der', 'bilateral', None
    requiere_preoperatorio: bool = True
    preoperatorio_completado: bool = False
    consentimiento_firmado: bool = False
    
    # DuraciÃ³n estimada personalizada (override del catÃ¡logo si procede)
    duracion_estimada_min: Optional[int] = None
    
    # Estado
    activa: bool = True
    cancelada: bool = False
    motivo_cancelacion: str = ""
    
    # Scores de priorizaciÃ³n (calculados)
    score_clinico: float = 0.0
    score_tiempo_espera: float = 0.0
    score_total: float = 0.0
    
    def __post_init__(self):
        """Calcula fecha lÃ­mite basada en prioridad"""
        if self.fecha_limite is None:
            dias_max = TIEMPOS_MAXIMOS_ESPERA.get(self.prioridad, 180)
            self.fecha_limite = self.fecha_indicacion + timedelta(days=dias_max)
    
    @property
    def dias_en_espera(self) -> int:
        """DÃ­as transcurridos desde la indicaciÃ³n"""
        return (date.today() - self.fecha_indicacion).days
    
    @property
    def dias_hasta_limite(self) -> int:
        """DÃ­as hasta la fecha lÃ­mite (negativo si pasada)"""
        return (self.fecha_limite - date.today()).days
    
    @property
    def porcentaje_tiempo_consumido(self) -> float:
        """Porcentaje del tiempo mÃ¡ximo de espera consumido"""
        dias_max = TIEMPOS_MAXIMOS_ESPERA.get(self.prioridad, 180)
        return min(self.dias_en_espera / dias_max, 1.5)  # Cap at 150%
    
    @property
    def esta_fuera_plazo(self) -> bool:
        """Indica si ha superado el tiempo mÃ¡ximo de espera"""
        return self.dias_hasta_limite < 0
    
    def duracion_esperada(self) -> int:
        """Retorna la duraciÃ³n estimada de la cirugÃ­a"""
        if self.duracion_estimada_min:
            return self.duracion_estimada_min
        return self.tipo_intervencion.duracion_media_min
    
    def duracion_con_variabilidad(self, seed: Optional[int] = None) -> int:
        """Genera una duraciÃ³n realista con variabilidad"""
        if seed:
            np.random.seed(seed)
        
        media = self.duracion_esperada()
        std = self.tipo_intervencion.duracion_std_min
        
        # DistribuciÃ³n log-normal para duraciones (siempre positivas, con cola a la derecha)
        duracion = np.random.lognormal(
            mean=np.log(media) - 0.5 * (std/media)**2,
            sigma=std/media
        )
        return max(int(duracion), 15)  # MÃ­nimo 15 minutos
    
    def calcular_score_clinico(self, fecha_referencia: date = None) -> float:
        """
        Calcula el score clÃ­nico basado en mÃºltiples factores.
        Retorna un valor entre 0 y 100.
        """
        if fecha_referencia is None:
            fecha_referencia = date.today()
        
        score = 0.0
        
        # 1. Prioridad CatSalut (30 puntos max)
        prioridad_scores = {
            PrioridadCatSalut.URGENTE: 30,
            PrioridadCatSalut.ONCOLOGICO_PRIORITARIO: 28,
            PrioridadCatSalut.ONCOLOGICO_ESTANDAR: 25,
            PrioridadCatSalut.CARDIACA: 22,
            PrioridadCatSalut.GARANTIZADO_180: 15,
            PrioridadCatSalut.REFERENCIA_P1: 18,
            PrioridadCatSalut.REFERENCIA_P2: 12,
            PrioridadCatSalut.REFERENCIA_P3: 8,
        }
        score += prioridad_scores.get(self.prioridad, 10)
        
        # 2. Tiempo en espera relativo (25 puntos max)
        pct_tiempo = self.porcentaje_tiempo_consumido
        if pct_tiempo >= 1.0:  # Fuera de plazo
            score += 25
        else:
            score += pct_tiempo * 20
        
        # 3. Complejidad y riesgo (15 puntos max)
        score += self.tipo_intervencion.complejidad * 2
        if self.tipo_intervencion.requiere_uci:
            score += 5
        
        # 4. Riesgo del paciente (15 puntos max)
        asa_score = {
            ClaseASA.ASA_I: 0,
            ClaseASA.ASA_II: 3,
            ClaseASA.ASA_III: 7,
            ClaseASA.ASA_IV: 12,
            ClaseASA.ASA_V: 15,
        }
        score += asa_score.get(self.paciente.clase_asa, 5)
        
        # 5. Factor edad (10 puntos max)
        score += self.paciente.factor_riesgo_edad() * 10
        
        # 6. Comorbilidades (5 puntos max)
        score += min(len(self.paciente.comorbilidades) * 1, 5)
        
        self.score_clinico = min(score, 100)
        return self.score_clinico


# =============================================================================
# MODELO DE CIRUGÃA PROGRAMADA (SLOT)
# =============================================================================

class EstadoCirugia(Enum):
    """Estado de una cirugÃ­a programada"""
    PROGRAMADA = auto()
    EN_CURSO = auto()
    COMPLETADA = auto()
    CANCELADA = auto()
    POSPUESTA = auto()


@dataclass
class CirugiaProgramada:
    """
    Una cirugÃ­a que ha sido asignada a un slot especÃ­fico
    en el programa quirÃºrgico.
    """
    id: str
    solicitud: SolicitudCirugia
    
    # AsignaciÃ³n temporal
    fecha: date
    hora_inicio: int  # Minutos desde medianoche
    duracion_programada_min: int
    
    # AsignaciÃ³n de recursos
    quirofano_id: int
    cirujano: Cirujano
    
    # Estado
    estado: EstadoCirugia = EstadoCirugia.PROGRAMADA
    
    # Tiempos reales (para seguimiento)
    hora_inicio_real: Optional[int] = None
    hora_fin_real: Optional[int] = None
    duracion_real_min: Optional[int] = None
    
    # InformaciÃ³n adicional
    posicion_en_quirofano: int = 0  # 1st, 2nd, 3rd... del dÃ­a
    notas_programacion: str = ""
    
    @property
    def hora_fin_programada(self) -> int:
        """Hora de fin programada en minutos desde medianoche"""
        return self.hora_inicio + self.duracion_programada_min
    
    @property
    def hora_inicio_str(self) -> str:
        """Hora de inicio en formato HH:MM"""
        h, m = divmod(self.hora_inicio, 60)
        return f"{h:02d}:{m:02d}"
    
    @property
    def hora_fin_str(self) -> str:
        """Hora de fin en formato HH:MM"""
        h, m = divmod(self.hora_fin_programada, 60)
        return f"{h:02d}:{m:02d}"
    
    def conflicto_con(self, otra: 'CirugiaProgramada', tiempo_limpieza: int = 30) -> bool:
        """Verifica si hay conflicto temporal con otra cirugÃ­a"""
        if self.fecha != otra.fecha or self.quirofano_id != otra.quirofano_id:
            return False
        
        fin_con_limpieza = self.hora_fin_programada + tiempo_limpieza
        return not (fin_con_limpieza <= otra.hora_inicio or 
                   otra.hora_fin_programada + tiempo_limpieza <= self.hora_inicio)


# =============================================================================
# MODELO DE PROGRAMA QUIRÃšRGICO DIARIO
# =============================================================================

@dataclass
class ProgramaDiario:
    """Programa quirÃºrgico para un dÃ­a especÃ­fico"""
    fecha: date
    cirugias: List[CirugiaProgramada] = field(default_factory=list)
    
    # MÃ©tricas del dÃ­a
    _metricas_calculadas: bool = False
    
    def agregar_cirugia(self, cirugia: CirugiaProgramada) -> bool:
        """Agrega una cirugÃ­a validando conflictos"""
        for c in self.cirugias:
            if cirugia.conflicto_con(c):
                return False
        self.cirugias.append(cirugia)
        self._metricas_calculadas = False
        return True
    
    def cirugias_por_quirofano(self, quirofano_id: int) -> List[CirugiaProgramada]:
        """Obtiene las cirugÃ­as de un quirÃ³fano ordenadas por hora"""
        return sorted(
            [c for c in self.cirugias if c.quirofano_id == quirofano_id],
            key=lambda x: x.hora_inicio
        )
    
    def utilizacion_quirofano(self, quirofano_id: int, 
                              hora_inicio: int = 480, hora_fin: int = 900) -> float:
        """
        Calcula el porcentaje de utilizaciÃ³n de un quirÃ³fano.
        hora_inicio/fin en minutos desde medianoche (default 8:00-15:00)
        """
        tiempo_total = hora_fin - hora_inicio
        tiempo_ocupado = sum(
            c.duracion_programada_min 
            for c in self.cirugias_por_quirofano(quirofano_id)
        )
        return min(tiempo_ocupado / tiempo_total, 1.0)
    
    def overtime_quirofano(self, quirofano_id: int, hora_fin: int = 900) -> int:
        """Calcula los minutos de overtime para un quirÃ³fano"""
        cirugias = self.cirugias_por_quirofano(quirofano_id)
        if not cirugias:
            return 0
        ultima = cirugias[-1]
        return max(0, ultima.hora_fin_programada - hora_fin)
    
    def total_cirugias(self) -> int:
        return len(self.cirugias)
    
    def cirugias_oncologicas(self) -> int:
        return sum(
            1 for c in self.cirugias 
            if c.solicitud.prioridad in [
                PrioridadCatSalut.ONCOLOGICO_PRIORITARIO,
                PrioridadCatSalut.ONCOLOGICO_ESTANDAR
            ]
        )
    
    def ingresos_uci_esperados(self) -> float:
        """Estima el nÃºmero de ingresos UCI esperados"""
        return sum(
            c.solicitud.tipo_intervencion.probabilidad_uci
            for c in self.cirugias
        )
    
    def resumen(self) -> Dict[str, Any]:
        """Genera un resumen del programa diario"""
        quirofanos_usados = set(c.quirofano_id for c in self.cirugias)
        
        return {
            'fecha': self.fecha.isoformat(),
            'total_cirugias': self.total_cirugias(),
            'cirugias_oncologicas': self.cirugias_oncologicas(),
            'quirofanos_activos': len(quirofanos_usados),
            'ingresos_uci_esperados': round(self.ingresos_uci_esperados(), 1),
            'utilizacion_media': np.mean([
                self.utilizacion_quirofano(q) for q in quirofanos_usados
            ]) if quirofanos_usados else 0,
            'overtime_total': sum(
                self.overtime_quirofano(q) for q in quirofanos_usados
            ),
        }


# =============================================================================
# MODELO DE PROGRAMA SEMANAL/PERIODO
# =============================================================================

@dataclass
class ProgramaPeriodo:
    """Programa quirÃºrgico para un periodo (tÃ­picamente 1-2 semanas)"""
    fecha_inicio: date
    fecha_fin: date
    programas_diarios: Dict[date, ProgramaDiario] = field(default_factory=dict)
    
    # Metadata de optimizaciÃ³n
    score_optimizacion: float = 0.0
    iteraciones_optimizador: int = 0
    tiempo_optimizacion_seg: float = 0.0
    restricciones_violadas: List[str] = field(default_factory=list)
    
    def obtener_dia(self, fecha: date) -> ProgramaDiario:
        """Obtiene o crea el programa de un dÃ­a"""
        if fecha not in self.programas_diarios:
            self.programas_diarios[fecha] = ProgramaDiario(fecha=fecha)
        return self.programas_diarios[fecha]
    
    def todas_las_cirugias(self) -> List[CirugiaProgramada]:
        """Retorna todas las cirugÃ­as del periodo"""
        cirugias = []
        for programa in self.programas_diarios.values():
            cirugias.extend(programa.cirugias)
        return cirugias
    
    def estadisticas_periodo(self) -> Dict[str, Any]:
        """Genera estadÃ­sticas del periodo completo"""
        todas = self.todas_las_cirugias()
        
        if not todas:
            return {'mensaje': 'No hay cirugÃ­as programadas'}
        
        # Conteos por prioridad
        por_prioridad = {}
        for p in PrioridadCatSalut:
            count = sum(1 for c in todas if c.solicitud.prioridad == p)
            if count > 0:
                por_prioridad[p.name] = count
        
        # Conteos por especialidad
        por_especialidad = {}
        for e in Especialidad:
            count = sum(1 for c in todas 
                       if c.solicitud.tipo_intervencion.especialidad == e)
            if count > 0:
                por_especialidad[e.name] = count
        
        # Utilizaciones
        utilizaciones = []
        for prog in self.programas_diarios.values():
            for q in range(1, 9):
                util = prog.utilizacion_quirofano(q)
                if util > 0:
                    utilizaciones.append(util)
        
        return {
            'periodo': f"{self.fecha_inicio} a {self.fecha_fin}",
            'dias_programados': len(self.programas_diarios),
            'total_cirugias': len(todas),
            'por_prioridad': por_prioridad,
            'por_especialidad': por_especialidad,
            'utilizacion_media': np.mean(utilizaciones) if utilizaciones else 0,
            'utilizacion_min': min(utilizaciones) if utilizaciones else 0,
            'utilizacion_max': max(utilizaciones) if utilizaciones else 0,
            'score_optimizacion': self.score_optimizacion,
        }


# =============================================================================
# MODELO DE RESTRICCIÃ“N APRENDIDA
# =============================================================================

@dataclass
class RestriccionAprendida:
    """
    RestricciÃ³n descubierta automÃ¡ticamente del anÃ¡lisis de datos histÃ³ricos.
    """
    id: str
    tipo: str  # 'secuencia', 'preferencia', 'coocurrencia', 'temporal', 'recurso'
    descripcion: str
    
    # Entidades involucradas
    entidades: Dict[str, Any] = field(default_factory=dict)
    
    # MÃ©tricas de la restricciÃ³n
    soporte: float = 0.0  # ProporciÃ³n de casos donde se cumple
    confianza: float = 0.0  # Fiabilidad de la restricciÃ³n
    lift: float = 1.0  # Mejora sobre el azar
    
    # Validez
    fecha_descubrimiento: date = field(default_factory=date.today)
    activa: bool = True
    validada_por_usuario: bool = False
    
    # Impacto
    penalizacion_incumplimiento: float = 1.0  # Factor de penalizaciÃ³n si se viola
    es_hard_constraint: bool = False  # Si es True, no se puede violar
    
    def aplica_a(self, cirugia: CirugiaProgramada) -> bool:
        """Determina si esta restricciÃ³n aplica a una cirugÃ­a dada"""
        # ImplementaciÃ³n especÃ­fica segÃºn tipo
        return True
    
    def se_cumple(self, cirugia: CirugiaProgramada, 
                  contexto: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verifica si la restricciÃ³n se cumple para una cirugÃ­a en un contexto.
        Retorna (cumple, mensaje)
        """
        # ImplementaciÃ³n especÃ­fica segÃºn tipo
        return True, ""


# =============================================================================
# UTILIDADES DE GENERACIÃ“N DE IDs
# =============================================================================

def generar_id() -> str:
    """Genera un ID Ãºnico"""
    return str(uuid.uuid4())[:8]


def generar_id_cirugia() -> str:
    """Genera un ID para cirugÃ­a con prefijo"""
    return f"CIR-{generar_id()}"


def generar_id_paciente() -> str:
    """Genera un ID para paciente con prefijo"""
    return f"PAC-{generar_id()}"


def generar_id_solicitud() -> str:
    """Genera un ID para solicitud con prefijo"""
    return f"SOL-{generar_id()}"
