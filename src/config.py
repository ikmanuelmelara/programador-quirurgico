"""
ConfiguraciÃ³n del Programador QuirÃºrgico Inteligente
=====================================================
Sistema de optimizaciÃ³n para bloque quirÃºrgico de cirugÃ­a general (8 quirÃ³fanos)
Basado en criterios de priorizaciÃ³n del CatSalut (Catalunya)

Autor: Sistema de IA para GestiÃ³n Sanitaria
VersiÃ³n: 1.0
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional
import numpy as np

# =============================================================================
# CRITERIOS DE PRIORIZACIÃ“N CATSALUT
# =============================================================================

class PrioridadCatSalut(Enum):
    """
    Niveles de prioridad segÃºn normativa CatSalut
    Basado en Ordre SLT/102/2015 y actualizaciones
    """
    # CirugÃ­a oncolÃ³gica - Tiempo garantizado
    ONCOLOGICO_PRIORITARIO = 1      # 45 dÃ­as mÃ¡ximo
    ONCOLOGICO_ESTANDAR = 2         # 60 dÃ­as mÃ¡ximo
    
    # CirugÃ­a cardÃ­aca - Tiempo garantizado
    CARDIACA = 3                     # 90 dÃ­as mÃ¡ximo
    
    # CirugÃ­a con tiempo garantizado largo
    GARANTIZADO_180 = 4              # 180 dÃ­as (cataratas, prÃ³tesis)
    
    # CirugÃ­a con tiempo de referencia (no garantizado pero orientativo)
    REFERENCIA_P1 = 5                # 90 dÃ­as referencia - Alta prioridad
    REFERENCIA_P2 = 6                # 180 dÃ­as referencia - Media prioridad  
    REFERENCIA_P3 = 7                # 365 dÃ­as referencia - Baja prioridad
    
    # Urgencias (fuera de programaciÃ³n electiva normal)
    URGENTE = 0


# Tiempos mÃ¡ximos de espera en dÃ­as segÃºn prioridad CatSalut
TIEMPOS_MAXIMOS_ESPERA = {
    PrioridadCatSalut.URGENTE: 1,
    PrioridadCatSalut.ONCOLOGICO_PRIORITARIO: 45,
    PrioridadCatSalut.ONCOLOGICO_ESTANDAR: 60,
    PrioridadCatSalut.CARDIACA: 90,
    PrioridadCatSalut.GARANTIZADO_180: 180,
    PrioridadCatSalut.REFERENCIA_P1: 90,
    PrioridadCatSalut.REFERENCIA_P2: 180,
    PrioridadCatSalut.REFERENCIA_P3: 365,
}


# =============================================================================
# ESPECIALIDADES EN BLOQUE DE CIRUGÃA GENERAL
# =============================================================================

class Especialidad(Enum):
    """Especialidades que operan en un bloque de cirugÃ­a general tÃ­pico"""
    CIRUGIA_GENERAL = auto()
    CIRUGIA_DIGESTIVA = auto()
    CIRUGIA_HEPATOBILIAR = auto()
    CIRUGIA_COLORRECTAL = auto()
    CIRUGIA_ENDOCRINA = auto()
    CIRUGIA_MAMA = auto()
    CIRUGIA_BARIATRICA = auto()
    UROLOGIA = auto()
    GINECOLOGIA = auto()
    CIRUGIA_VASCULAR = auto()
    TRAUMATOLOGIA = auto()  # CirugÃ­as menores/ambulatorias
    CIRUGIA_PLASTICA = auto()
    ORL = auto()
    OFTALMOLOGIA = auto()


# Colores para visualizaciÃ³n
COLORES_ESPECIALIDAD = {
    Especialidad.CIRUGIA_GENERAL: "#3498db",
    Especialidad.CIRUGIA_DIGESTIVA: "#2ecc71",
    Especialidad.CIRUGIA_HEPATOBILIAR: "#9b59b6",
    Especialidad.CIRUGIA_COLORRECTAL: "#e74c3c",
    Especialidad.CIRUGIA_ENDOCRINA: "#f39c12",
    Especialidad.CIRUGIA_MAMA: "#e91e63",
    Especialidad.CIRUGIA_BARIATRICA: "#00bcd4",
    Especialidad.UROLOGIA: "#ff9800",
    Especialidad.GINECOLOGIA: "#ff5722",
    Especialidad.CIRUGIA_VASCULAR: "#795548",
    Especialidad.TRAUMATOLOGIA: "#607d8b",
    Especialidad.CIRUGIA_PLASTICA: "#9c27b0",
    Especialidad.ORL: "#4caf50",
    Especialidad.OFTALMOLOGIA: "#03a9f4",
}


# =============================================================================
# TIPOS DE INTERVENCIÃ“N CON DURACIÃ“N ESTIMADA
# =============================================================================

@dataclass
class TipoIntervencion:
    """DefiniciÃ³n de un tipo de intervenciÃ³n quirÃºrgica"""
    codigo: str
    nombre: str
    especialidad: Especialidad
    duracion_media_min: int
    duracion_std_min: int
    requiere_uci: bool = False
    probabilidad_uci: float = 0.0
    estancia_uci_media_dias: float = 0.0
    estancia_hospital_media_dias: float = 1.0
    complejidad: int = 1  # 1-5
    requiere_equipo_especial: List[str] = field(default_factory=list)
    prioridad_tipica: PrioridadCatSalut = PrioridadCatSalut.REFERENCIA_P2


# CatÃ¡logo de intervenciones (muestra representativa)
CATALOGO_INTERVENCIONES: Dict[str, TipoIntervencion] = {
    # CIRUGÃA GENERAL / DIGESTIVA
    "COL_LAP": TipoIntervencion(
        "COL_LAP", "ColecistectomÃ­a laparoscÃ³pica", 
        Especialidad.CIRUGIA_DIGESTIVA, 60, 15,
        estancia_hospital_media_dias=1, complejidad=2
    ),
    "COL_ABR": TipoIntervencion(
        "COL_ABR", "ColecistectomÃ­a abierta",
        Especialidad.CIRUGIA_DIGESTIVA, 90, 20,
        estancia_hospital_media_dias=3, complejidad=3
    ),
    "HERN_ING": TipoIntervencion(
        "HERN_ING", "Hernioplastia inguinal",
        Especialidad.CIRUGIA_GENERAL, 45, 10,
        estancia_hospital_media_dias=0.5, complejidad=1
    ),
    "HERN_UMB": TipoIntervencion(
        "HERN_UMB", "Hernioplastia umbilical",
        Especialidad.CIRUGIA_GENERAL, 40, 10,
        estancia_hospital_media_dias=0.5, complejidad=1
    ),
    "HERN_INC": TipoIntervencion(
        "HERN_INC", "Hernioplastia incisional/eventraciÃ³n",
        Especialidad.CIRUGIA_GENERAL, 120, 30,
        estancia_hospital_media_dias=3, complejidad=3
    ),
    "APEND_LAP": TipoIntervencion(
        "APEND_LAP", "ApendicectomÃ­a laparoscÃ³pica",
        Especialidad.CIRUGIA_GENERAL, 50, 15,
        estancia_hospital_media_dias=1, complejidad=2,
        prioridad_tipica=PrioridadCatSalut.REFERENCIA_P1
    ),
    "COLECT_DER": TipoIntervencion(
        "COLECT_DER", "HemicolectomÃ­a derecha",
        Especialidad.CIRUGIA_COLORRECTAL, 180, 40,
        requiere_uci=True, probabilidad_uci=0.15,
        estancia_uci_media_dias=1, estancia_hospital_media_dias=7,
        complejidad=4, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "COLECT_IZQ": TipoIntervencion(
        "COLECT_IZQ", "HemicolectomÃ­a izquierda",
        Especialidad.CIRUGIA_COLORRECTAL, 200, 45,
        requiere_uci=True, probabilidad_uci=0.2,
        estancia_uci_media_dias=1, estancia_hospital_media_dias=8,
        complejidad=4, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "RECT_ANT": TipoIntervencion(
        "RECT_ANT", "ResecciÃ³n anterior de recto",
        Especialidad.CIRUGIA_COLORRECTAL, 240, 50,
        requiere_uci=True, probabilidad_uci=0.3,
        estancia_uci_media_dias=2, estancia_hospital_media_dias=10,
        complejidad=5, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "AAP": TipoIntervencion(
        "AAP", "AmputaciÃ³n abdominoperineal",
        Especialidad.CIRUGIA_COLORRECTAL, 300, 60,
        requiere_uci=True, probabilidad_uci=0.4,
        estancia_uci_media_dias=2, estancia_hospital_media_dias=12,
        complejidad=5, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "GASTRECT": TipoIntervencion(
        "GASTRECT", "GastrectomÃ­a",
        Especialidad.CIRUGIA_DIGESTIVA, 240, 50,
        requiere_uci=True, probabilidad_uci=0.35,
        estancia_uci_media_dias=2, estancia_hospital_media_dias=10,
        complejidad=5, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "ESOFAG": TipoIntervencion(
        "ESOFAG", "EsofagectomÃ­a",
        Especialidad.CIRUGIA_DIGESTIVA, 360, 70,
        requiere_uci=True, probabilidad_uci=0.8,
        estancia_uci_media_dias=4, estancia_hospital_media_dias=15,
        complejidad=5, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO,
        requiere_equipo_especial=["toracoscopia"]
    ),
    "HEPAT_SEG": TipoIntervencion(
        "HEPAT_SEG", "HepatectomÃ­a segmentaria",
        Especialidad.CIRUGIA_HEPATOBILIAR, 180, 40,
        requiere_uci=True, probabilidad_uci=0.3,
        estancia_uci_media_dias=1, estancia_hospital_media_dias=7,
        complejidad=4, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "HEPAT_MAY": TipoIntervencion(
        "HEPAT_MAY", "HepatectomÃ­a mayor",
        Especialidad.CIRUGIA_HEPATOBILIAR, 300, 60,
        requiere_uci=True, probabilidad_uci=0.6,
        estancia_uci_media_dias=3, estancia_hospital_media_dias=12,
        complejidad=5, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "WHIPPLE": TipoIntervencion(
        "WHIPPLE", "DuodenopancreatectomÃ­a cefÃ¡lica",
        Especialidad.CIRUGIA_HEPATOBILIAR, 420, 80,
        requiere_uci=True, probabilidad_uci=0.7,
        estancia_uci_media_dias=4, estancia_hospital_media_dias=18,
        complejidad=5, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO,
        requiere_equipo_especial=["ecografo_intraop"]
    ),
    
    # CIRUGÃA DE MAMA
    "MAST_SIMPLE": TipoIntervencion(
        "MAST_SIMPLE", "MastectomÃ­a simple",
        Especialidad.CIRUGIA_MAMA, 90, 20,
        estancia_hospital_media_dias=2, complejidad=3,
        prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "MAST_RAD": TipoIntervencion(
        "MAST_RAD", "MastectomÃ­a radical modificada",
        Especialidad.CIRUGIA_MAMA, 150, 30,
        estancia_hospital_media_dias=3, complejidad=4,
        prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "TUMORECT": TipoIntervencion(
        "TUMORECT", "TumorectomÃ­a/cuadrantectomÃ­a",
        Especialidad.CIRUGIA_MAMA, 60, 15,
        estancia_hospital_media_dias=1, complejidad=2,
        prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "BSGC": TipoIntervencion(
        "BSGC", "Biopsia selectiva ganglio centinela",
        Especialidad.CIRUGIA_MAMA, 45, 15,
        estancia_hospital_media_dias=0.5, complejidad=2,
        prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO,
        requiere_equipo_especial=["gamma_probe"]
    ),
    
    # CIRUGÃA ENDOCRINA
    "TIROID_TOTAL": TipoIntervencion(
        "TIROID_TOTAL", "TiroidectomÃ­a total",
        Especialidad.CIRUGIA_ENDOCRINA, 120, 25,
        estancia_hospital_media_dias=2, complejidad=3,
        requiere_equipo_especial=["neuromonitor"]
    ),
    "TIROID_PARC": TipoIntervencion(
        "TIROID_PARC", "HemitiroidectomÃ­a",
        Especialidad.CIRUGIA_ENDOCRINA, 90, 20,
        estancia_hospital_media_dias=1, complejidad=2,
        requiere_equipo_especial=["neuromonitor"]
    ),
    "PARATIR": TipoIntervencion(
        "PARATIR", "ParatiroidectomÃ­a",
        Especialidad.CIRUGIA_ENDOCRINA, 90, 25,
        estancia_hospital_media_dias=1, complejidad=3,
        requiere_equipo_especial=["gamma_probe"]
    ),
    "ADRENAL": TipoIntervencion(
        "ADRENAL", "AdrenalectomÃ­a laparoscÃ³pica",
        Especialidad.CIRUGIA_ENDOCRINA, 150, 35,
        estancia_hospital_media_dias=3, complejidad=4
    ),
    
    # CIRUGÃA BARIÃTRICA
    "BYPASS_G": TipoIntervencion(
        "BYPASS_G", "Bypass gÃ¡strico",
        Especialidad.CIRUGIA_BARIATRICA, 180, 40,
        estancia_hospital_media_dias=3, complejidad=4
    ),
    "SLEEVE": TipoIntervencion(
        "SLEEVE", "GastrectomÃ­a vertical (sleeve)",
        Especialidad.CIRUGIA_BARIATRICA, 120, 30,
        estancia_hospital_media_dias=2, complejidad=3
    ),
    "BANDA_G": TipoIntervencion(
        "BANDA_G", "Banda gÃ¡strica ajustable",
        Especialidad.CIRUGIA_BARIATRICA, 90, 20,
        estancia_hospital_media_dias=1, complejidad=2
    ),
    
    # UROLOGÃA
    "PROST_RAD": TipoIntervencion(
        "PROST_RAD", "ProstatectomÃ­a radical",
        Especialidad.UROLOGIA, 180, 40,
        estancia_hospital_media_dias=5, complejidad=4,
        prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_ESTANDAR
    ),
    "RTU_PROST": TipoIntervencion(
        "RTU_PROST", "RTU prÃ³stata",
        Especialidad.UROLOGIA, 60, 20,
        estancia_hospital_media_dias=2, complejidad=2
    ),
    "RTU_VES": TipoIntervencion(
        "RTU_VES", "RTU vesical",
        Especialidad.UROLOGIA, 45, 15,
        estancia_hospital_media_dias=1, complejidad=2,
        prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_ESTANDAR
    ),
    "NEFR_RAD": TipoIntervencion(
        "NEFR_RAD", "NefrectomÃ­a radical",
        Especialidad.UROLOGIA, 180, 40,
        requiere_uci=True, probabilidad_uci=0.15,
        estancia_hospital_media_dias=5, complejidad=4,
        prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "NEFR_PARC": TipoIntervencion(
        "NEFR_PARC", "NefrectomÃ­a parcial",
        Especialidad.UROLOGIA, 150, 35,
        estancia_hospital_media_dias=4, complejidad=4,
        prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_PRIORITARIO
    ),
    "CISTECT": TipoIntervencion(
        "CISTECT", "CistectomÃ­a radical",
        Especialidad.UROLOGIA, 300, 60,
        requiere_uci=True, probabilidad_uci=0.4,
        estancia_uci_media_dias=2, estancia_hospital_media_dias=12,
        complejidad=5, prioridad_tipica=PrioridadCatSalut.ONCOLOGICO_ESTANDAR
    ),
    
    # GINECOLOGÃA
    "HIST_ABD": TipoIntervencion(
        "HIST_ABD", "HisterectomÃ­a abdominal",
        Especialidad.GINECOLOGIA, 120, 30,
        estancia_hospital_media_dias=3, complejidad=3
    ),
    "HIST_LAP": TipoIntervencion(
        "HIST_LAP", "HisterectomÃ­a laparoscÃ³pica",
        Especialidad.GINECOLOGIA, 150, 35,
        estancia_hospital_media_dias=2, complejidad=3
    ),
    "HIST_VAG": TipoIntervencion(
        "HIST_VAG", "HisterectomÃ­a vaginal",
        Especialidad.GINECOLOGIA, 90, 25,
        estancia_hospital_media_dias=2, complejidad=3
    ),
    "OOFOR": TipoIntervencion(
        "OOFOR", "OoforectomÃ­a",
        Especialidad.GINECOLOGIA, 60, 20,
        estancia_hospital_media_dias=1, complejidad=2
    ),
    "MIOM": TipoIntervencion(
        "MIOM", "MiomectomÃ­a",
        Especialidad.GINECOLOGIA, 90, 30,
        estancia_hospital_media_dias=2, complejidad=3
    ),
    
    # CIRUGÃA VASCULAR
    "VARIC": TipoIntervencion(
        "VARIC", "SafenectomÃ­a/cirugÃ­a varices",
        Especialidad.CIRUGIA_VASCULAR, 60, 20,
        estancia_hospital_media_dias=0.5, complejidad=2
    ),
    "EVAR": TipoIntervencion(
        "EVAR", "EVAR (endoprÃ³tesis aÃ³rtica)",
        Especialidad.CIRUGIA_VASCULAR, 180, 45,
        requiere_uci=True, probabilidad_uci=0.5,
        estancia_uci_media_dias=1, estancia_hospital_media_dias=4,
        complejidad=5, requiere_equipo_especial=["arco_c", "endoprotesis"]
    ),
    "BYPASS_FEM": TipoIntervencion(
        "BYPASS_FEM", "Bypass femoropoplÃ­teo",
        Especialidad.CIRUGIA_VASCULAR, 180, 40,
        estancia_hospital_media_dias=5, complejidad=4
    ),
    "ENDART": TipoIntervencion(
        "ENDART", "EndarterectomÃ­a carotÃ­dea",
        Especialidad.CIRUGIA_VASCULAR, 120, 30,
        requiere_uci=True, probabilidad_uci=0.3,
        estancia_uci_media_dias=1, estancia_hospital_media_dias=3,
        complejidad=4
    ),
    
    # OTROS (cirugÃ­a menor/ambulatoria)
    "EXCISION": TipoIntervencion(
        "EXCISION", "ExÃ©resis lesiÃ³n cutÃ¡nea",
        Especialidad.CIRUGIA_PLASTICA, 30, 10,
        estancia_hospital_media_dias=0, complejidad=1
    ),
    "PILONIDAL": TipoIntervencion(
        "PILONIDAL", "Quiste pilonidal",
        Especialidad.CIRUGIA_GENERAL, 45, 15,
        estancia_hospital_media_dias=0.5, complejidad=1
    ),
    "FISTULA_AN": TipoIntervencion(
        "FISTULA_AN", "FÃ­stula anal",
        Especialidad.CIRUGIA_COLORRECTAL, 45, 15,
        estancia_hospital_media_dias=0.5, complejidad=2
    ),
    "HEMORR": TipoIntervencion(
        "HEMORR", "HemorroidectomÃ­a",
        Especialidad.CIRUGIA_COLORRECTAL, 40, 10,
        estancia_hospital_media_dias=1, complejidad=2
    ),
}


# =============================================================================
# CONFIGURACIÃ“N DE QUIRÃ“FANOS
# =============================================================================

@dataclass
class Quirofano:
    """ConfiguraciÃ³n de un quirÃ³fano"""
    id: int
    nombre: str
    especialidades_permitidas: List[Especialidad]
    equipamiento_especial: List[str] = field(default_factory=list)
    horario_inicio: int = 8 * 60  # 08:00 en minutos
    horario_fin: int = 15 * 60    # 15:00 en minutos (7h jornada)
    activo: bool = True


# ConfiguraciÃ³n por defecto de 8 quirÃ³fanos
QUIROFANOS_DEFAULT = [
    Quirofano(
        1, "QuirÃ³fano 1 - CirugÃ­a Mayor Digestiva",
        [Especialidad.CIRUGIA_DIGESTIVA, Especialidad.CIRUGIA_HEPATOBILIAR,
         Especialidad.CIRUGIA_COLORRECTAL],
        ["laparoscopia_avanzada", "ecografo_intraop"]
    ),
    Quirofano(
        2, "QuirÃ³fano 2 - CirugÃ­a Mayor Digestiva",
        [Especialidad.CIRUGIA_DIGESTIVA, Especialidad.CIRUGIA_HEPATOBILIAR,
         Especialidad.CIRUGIA_COLORRECTAL],
        ["laparoscopia_avanzada", "toracoscopia"]
    ),
    Quirofano(
        3, "QuirÃ³fano 3 - CirugÃ­a General y Mama",
        [Especialidad.CIRUGIA_GENERAL, Especialidad.CIRUGIA_MAMA,
         Especialidad.CIRUGIA_ENDOCRINA],
        ["gamma_probe", "neuromonitor"]
    ),
    Quirofano(
        4, "QuirÃ³fano 4 - UrologÃ­a",
        [Especialidad.UROLOGIA],
        ["laparoscopia_avanzada", "robot_opcional"]
    ),
    Quirofano(
        5, "QuirÃ³fano 5 - GinecologÃ­a",
        [Especialidad.GINECOLOGIA],
        ["laparoscopia_avanzada", "histeroscopia"]
    ),
    Quirofano(
        6, "QuirÃ³fano 6 - CirugÃ­a BariÃ¡trica y General",
        [Especialidad.CIRUGIA_BARIATRICA, Especialidad.CIRUGIA_GENERAL],
        ["laparoscopia_avanzada"]
    ),
    Quirofano(
        7, "QuirÃ³fano 7 - CirugÃ­a Vascular",
        [Especialidad.CIRUGIA_VASCULAR],
        ["arco_c", "endoprotesis"]
    ),
    Quirofano(
        8, "QuirÃ³fano 8 - CMA (CirugÃ­a Mayor Ambulatoria)",
        [Especialidad.CIRUGIA_GENERAL, Especialidad.CIRUGIA_PLASTICA,
         Especialidad.CIRUGIA_COLORRECTAL, Especialidad.UROLOGIA],
        []
    ),
]


# =============================================================================
# RESTRICCIONES COMUNES EN LA LITERATURA
# =============================================================================

@dataclass
class RestriccionesGlobales:
    """
    Restricciones basadas en la literatura cientÃ­fica sobre programaciÃ³n quirÃºrgica
    Referencias: 
    - Cardoen et al. (2010) - Operating room planning and scheduling
    - Guerriero & Guido (2011) - Operational research in healthcare
    - Zhu et al. (2019) - OR scheduling review
    """
    
    # RESTRICCIONES DE RECURSOS MATERIALES
    max_cirugias_simultaneas: int = 8  # NÃºmero de quirÃ³fanos
    camas_uci_disponibles: int = 6
    camas_reanimacion: int = 4
    camas_hospitalizacion: int = 40
    
    # RESTRICCIONES DE RECURSOS HUMANOS
    max_anestesiologos_turno: int = 10
    max_enfermeras_quirofano_turno: int = 20
    ratio_enfermera_quirofano: int = 2  # 2 enfermeras por quirÃ³fano activo
    
    # RESTRICCIONES TEMPORALES
    tiempo_limpieza_entre_cirugias_min: int = 30
    tiempo_limpieza_contaminada_min: int = 45
    tiempo_setup_quirofano_min: int = 15
    margen_seguridad_jornada_min: int = 30  # No programar si termina muy justo
    
    # RESTRICCIONES DE SECUENCIACIÃ“N
    oncologico_primero_manana: bool = True  # CirugÃ­a oncolÃ³gica a primera hora
    evitar_contaminada_antes_limpia: bool = True
    
    # HOLGURAS Y BUFFERS
    buffer_tiempo_porcentaje: float = 0.1  # 10% extra sobre tiempo estimado
    max_overtime_permitido_min: int = 60
    
    # RESTRICCIONES DE CAPACIDAD UCI
    max_ingresos_uci_dia: int = 4
    ocupacion_uci_umbral_alerta: float = 0.8


# =============================================================================
# CRITERIOS DE PRIORIZACIÃ“N MULTI-DIMENSIONAL
# =============================================================================

@dataclass
class PesosOptimizacion:
    """
    Pesos configurables para balancear prioridad clÃ­nica vs eficiencia operativa
    Los usuarios pueden ajustar estos valores segÃºn sus preferencias
    
    Basado en criterios identificados en la literatura (AIAQS 2010):
    - Gravedad de la enfermedad y repercusiones
    - Riesgo asociado a la demora
    - Efectividad clÃ­nica esperada
    - Tiempo en lista de espera
    - Impacto en calidad de vida
    """
    
    # === BALANCE PRINCIPAL (debe sumar 1.0) ===
    peso_prioridad_clinica: float = 0.6
    peso_eficiencia_operativa: float = 0.4
    
    # === SUB-CRITERIOS CLÃNICOS (deben sumar 1.0) ===
    peso_gravedad_enfermedad: float = 0.30
    peso_riesgo_demora: float = 0.25
    peso_tiempo_espera_relativo: float = 0.25  # Tiempo esperado / tiempo mÃ¡ximo
    peso_efectividad_esperada: float = 0.10
    peso_impacto_calidad_vida: float = 0.10
    
    # === SUB-CRITERIOS EFICIENCIA (deben sumar 1.0) ===
    peso_utilizacion_quirofano: float = 0.35
    peso_minimizar_overtime: float = 0.25
    peso_balanceo_carga: float = 0.20
    peso_agrupacion_especialidad: float = 0.10
    peso_minimizar_gaps: float = 0.10
    
    def validar(self) -> bool:
        """Valida que los pesos estÃ©n correctamente configurados"""
        # Balance principal
        if abs(self.peso_prioridad_clinica + self.peso_eficiencia_operativa - 1.0) > 0.001:
            return False
        
        # Sub-criterios clÃ­nicos
        suma_clinica = (self.peso_gravedad_enfermedad + self.peso_riesgo_demora +
                       self.peso_tiempo_espera_relativo + self.peso_efectividad_esperada +
                       self.peso_impacto_calidad_vida)
        if abs(suma_clinica - 1.0) > 0.001:
            return False
        
        # Sub-criterios eficiencia
        suma_eficiencia = (self.peso_utilizacion_quirofano + self.peso_minimizar_overtime +
                         self.peso_balanceo_carga + self.peso_agrupacion_especialidad +
                         self.peso_minimizar_gaps)
        if abs(suma_eficiencia - 1.0) > 0.001:
            return False
        
        return True


# Configuraciones predefinidas
PESOS_CLINICO_PRIORITARIO = PesosOptimizacion(
    peso_prioridad_clinica=0.8,
    peso_eficiencia_operativa=0.2
)

PESOS_EFICIENCIA_PRIORITARIA = PesosOptimizacion(
    peso_prioridad_clinica=0.4,
    peso_eficiencia_operativa=0.6
)

PESOS_BALANCEADO = PesosOptimizacion()  # Default 60/40


# =============================================================================
# CONFIGURACIÃ“N DE APRENDIZAJE DE RESTRICCIONES
# =============================================================================

@dataclass
class ConfigAprendizaje:
    """ConfiguraciÃ³n para el mÃ³dulo de aprendizaje automÃ¡tico de restricciones"""
    
    # ParÃ¡metros de detecciÃ³n de patrones
    min_soporte_regla: float = 0.1  # MÃ­nimo 10% de casos para considerar patrÃ³n
    min_confianza_regla: float = 0.7  # 70% de confianza mÃ­nima
    
    # Ventana temporal para anÃ¡lisis
    dias_historico_analisis: int = 365
    min_casos_para_aprender: int = 50
    
    # Umbrales de detecciÃ³n de restricciones implÃ­citas
    umbral_coocurrencia_cirujano_tipo: float = 0.8
    umbral_preferencia_quirofano: float = 0.7
    umbral_secuencia_temporal: float = 0.6
    
    # ActualizaciÃ³n del modelo
    frecuencia_reentrenamiento_dias: int = 30
    

# =============================================================================
# CONFIGURACIÃ“N GENERAL DEL SISTEMA
# =============================================================================

@dataclass
class ConfiguracionSistema:
    """ConfiguraciÃ³n general del sistema de programaciÃ³n"""
    
    # Horizonte de planificaciÃ³n
    horizonte_dias: int = 14  # Planificar 2 semanas
    
    # DÃ­as laborables (0=Lunes, 6=Domingo)
    dias_operativos: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # L-V
    
    # Festivos (lista de fechas en formato string 'YYYY-MM-DD')
    festivos: List[str] = field(default_factory=list)
    
    # Semilla para reproducibilidad
    random_seed: int = 42
    
    # ParÃ¡metros del optimizador
    max_iteraciones_optimizador: int = 1000
    tiempo_max_optimizacion_seg: int = 300
    
    # Verbosidad
    nivel_log: str = "INFO"  # DEBUG, INFO, WARNING, ERROR


# Instancia por defecto
CONFIG_DEFAULT = ConfiguracionSistema()
RESTRICCIONES_DEFAULT = RestriccionesGlobales()
PESOS_DEFAULT = PesosOptimizacion()
CONFIG_APRENDIZAJE_DEFAULT = ConfigAprendizaje()
