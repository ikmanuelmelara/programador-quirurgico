# =============================================================================
# PROGRAMADOR QUIRÚRGICO INTELIGENTE - VERSIÓN 4.5
# =============================================================================
# Sistema con ML Real para Google Colab
# Incluye: Sesiones M/T + Restricciones + Comparador + PREDICCIÓN URGENCIAS ML
#          + Reservas integradas + URGENCIAS DIFERIDAS + VISTA CALENDARIO
# =============================================================================

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from collections import defaultdict
import random
import time

# Importar sistema básico
from main import ProgramadorQuirurgico
from config import PrioridadCatSalut, TIEMPOS_MAXIMOS_ESPERA, QUIROFANOS_DEFAULT, Especialidad
from models import CirugiaProgramada, ProgramaDiario, ProgramaPeriodo, generar_id_cirugia

# Intentar importar módulos avanzados
try:
    from constraint_learning_advanced import AprendizajeRestriccionesAvanzado
    ML_AVANZADO = True
    print("✅ Módulo de ML avanzado cargado")
except ImportError as e:
    ML_AVANZADO = False
    print(f"⚠️ Usando módulo de restricciones básico: {e}")

try:
    from optimizer_advanced import OptimizadorAvanzado, ORTOOLS_AVAILABLE, DEAP_AVAILABLE
    OPT_AVANZADO = True
    print(f"✅ Optimizador avanzado cargado (OR-Tools: {ORTOOLS_AVAILABLE}, DEAP: {DEAP_AVAILABLE})")
except ImportError as e:
    OPT_AVANZADO = False
    ORTOOLS_AVAILABLE = False
    DEAP_AVAILABLE = False
    print(f"⚠️ Usando optimizador básico: {e}")

# Inicializar
print("\n🔄 Inicializando sistema...")
programador = ProgramadorQuirurgico(seed=42)
programador.inicializar_datos_sinteticos(n_solicitudes=500, dias_historico=365)

# Si tenemos ML avanzado, re-analizar restricciones
restricciones_ml = []
if ML_AVANZADO:
    print("\n🤖 Analizando restricciones con ML avanzado...")
    aprendizaje_avanzado = AprendizajeRestriccionesAvanzado()
    restricciones_ml = aprendizaje_avanzado.analizar_historico(programador.historico)
    print(f"✅ {len(restricciones_ml)} restricciones descubiertas con ML")

# Inicializar predictor de urgencias
PREDICTOR_DISPONIBLE = False
predictor_urgencias = None
urgencias_predictor = None  # Alias para compatibilidad
try:
    from urgencias_predictor import PredictorUrgencias, crear_predictor_desde_historico
    print("\n🔮 Entrenando predictor de urgencias...")
    predictor_urgencias = crear_predictor_desde_historico(programador.historico)
    urgencias_predictor = predictor_urgencias  # Alias
    PREDICTOR_DISPONIBLE = True
    print(f"✅ Predictor entrenado: {len(predictor_urgencias.modelos)} modelos")
except ImportError as e:
    print(f"⚠️ Predictor de urgencias no disponible: {e}")
except Exception as e:
    print(f"⚠️ Error entrenando predictor: {e}")

# Inicializar predictor de demanda
PREDICTOR_DEMANDA_DISPONIBLE = False
predictor_demanda = None
try:
    from predictor_demanda import (
        PredictorDemanda, GeneradorHistoricoMovimientos, 
        crear_predictor_desde_historico_cirugias
    )
    print("\n📈 Entrenando predictor de demanda...")
    predictor_demanda = crear_predictor_desde_historico_cirugias(programador.historico)
    predictor_demanda.entrenar()
    PREDICTOR_DEMANDA_DISPONIBLE = True
    print(f"✅ Predictor de demanda entrenado")
except ImportError as e:
    print(f"⚠️ Predictor de demanda no disponible: {e}")
except Exception as e:
    print(f"⚠️ Error entrenando predictor demanda: {e}")

# Inicializar simulador What-If
SIMULADOR_DISPONIBLE = False
simulador_whatif = None
try:
    from simulador_whatif import (
        SimuladorWhatIf, Escenario, TipoEscenario, 
        ResultadoSimulacion, crear_escenario_rapido
    )
    SIMULADOR_DISPONIBLE = True
    print(f"✅ Simulador What-If disponible")
except ImportError as e:
    print(f"⚠️ Simulador What-If no disponible: {e}")
except Exception as e:
    print(f"⚠️ Error cargando simulador: {e}")

print("\n✅ Sistema listo")


# =============================================================================
# CONFIGURACIÓN DE SESIONES QUIRÚRGICAS
# =============================================================================

ESPECIALIDADES_NOMBRES = {
    'CIRUGIA_GENERAL': 'Cir. General',
    'CIRUGIA_DIGESTIVA': 'Digestivo',
    'CIRUGIA_HEPATOBILIAR': 'Hepatobiliar',
    'CIRUGIA_COLORRECTAL': 'Colorrectal',
    'CIRUGIA_ENDOCRINA': 'Endocrina',
    'CIRUGIA_MAMA': 'Mama',
    'CIRUGIA_BARIATRICA': 'Bariátrica',
    'UROLOGIA': 'Urología',
    'GINECOLOGIA': 'Ginecología',
    'CIRUGIA_VASCULAR': 'Vascular',
    'CIRUGIA_PLASTICA': 'Plástica',
    'TRAUMATOLOGIA': 'Traumatología',
    'ORL': 'ORL',
    'OFTALMOLOGIA': 'Oftalmología',
    'LIBRE': '— Libre —',
    'CERRADO': '✗ Cerrado'
}

LISTA_ESPECIALIDADES = list(ESPECIALIDADES_NOMBRES.keys())
DIAS_SEMANA = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
TURNOS = ['Mañana', 'Tarde']
NUM_QUIROFANOS = 8

HORARIO_MANANA_INICIO = 8 * 60
HORARIO_MANANA_FIN = 15 * 60
HORARIO_TARDE_INICIO = 15 * 60
HORARIO_TARDE_FIN = 20 * 60

configuracion_sesiones = {}

def inicializar_configuracion_default():
    global configuracion_sesiones
    configuracion_sesiones = {}
    
    asignacion_default = {
        1: 'CIRUGIA_DIGESTIVA', 2: 'CIRUGIA_COLORRECTAL', 3: 'CIRUGIA_MAMA',
        4: 'UROLOGIA', 5: 'GINECOLOGIA', 6: 'CIRUGIA_BARIATRICA',
        7: 'CIRUGIA_VASCULAR', 8: 'CIRUGIA_GENERAL'
    }
    
    for q in range(1, NUM_QUIROFANOS + 1):
        configuracion_sesiones[q] = {}
        esp_default = asignacion_default.get(q, 'CIRUGIA_GENERAL')
        for dia in DIAS_SEMANA:
            configuracion_sesiones[q][dia] = {'Mañana': esp_default, 'Tarde': 'LIBRE'}
    
    return configuracion_sesiones

inicializar_configuracion_default()

# Variable global para almacenar resultado de optimización
ultimo_resultado_sesiones = None

# =============================================================================
# MODELO DE URGENCIAS DIFERIDAS
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class EstadoUrgencia(Enum):
    """Estado de una urgencia diferida"""
    PENDIENTE = "pendiente"
    PROGRAMADA = "programada"
    OPERADA = "operada"
    CANCELADA = "cancelada"

@dataclass
class UrgenciaDiferida:
    """Modelo para urgencias diferidas (24-72h)"""
    id: str
    paciente_nombre: str
    edad: int
    especialidad: str
    diagnostico: str
    procedimiento: str
    duracion_estimada_min: int
    fecha_entrada: date
    horas_limite: int  # Máximo horas para operar (24, 48, 72)
    prioridad: int  # 1=máxima, 2=alta, 3=media
    estado: EstadoUrgencia = EstadoUrgencia.PENDIENTE
    fecha_programada: Optional[date] = None
    quirofano_asignado: Optional[int] = None
    notas: str = ""
    
    @property
    def fecha_limite(self) -> date:
        """Calcula la fecha/hora límite"""
        return self.fecha_entrada + timedelta(hours=self.horas_limite)
    
    @property
    def horas_restantes(self) -> float:
        """Horas restantes hasta el límite"""
        from datetime import datetime
        ahora = datetime.now()
        limite = datetime.combine(self.fecha_limite, datetime.min.time())
        delta = limite - ahora
        return max(0, delta.total_seconds() / 3600)
    
    @property
    def es_critica(self) -> bool:
        """True si quedan menos de 12 horas"""
        return self.horas_restantes < 12

# Lista global de urgencias diferidas
urgencias_diferidas: list = []
urgencia_id_counter = 0

def generar_id_urgencia() -> str:
    """Genera un ID único para urgencia"""
    global urgencia_id_counter
    urgencia_id_counter += 1
    return f"URG-{urgencia_id_counter:04d}"

def agregar_urgencia(paciente: str, edad: int, especialidad: str, 
                     diagnostico: str, procedimiento: str, 
                     duracion: int, horas_limite: int, prioridad: int,
                     notas: str = "") -> UrgenciaDiferida:
    """Añade una nueva urgencia diferida a la lista"""
    urgencia = UrgenciaDiferida(
        id=generar_id_urgencia(),
        paciente_nombre=paciente,
        edad=edad,
        especialidad=especialidad,
        diagnostico=diagnostico,
        procedimiento=procedimiento,
        duracion_estimada_min=duracion,
        fecha_entrada=date.today(),
        horas_limite=horas_limite,
        prioridad=prioridad,
        notas=notas
    )
    urgencias_diferidas.append(urgencia)
    return urgencia

def obtener_urgencias_pendientes() -> list:
    """Retorna urgencias pendientes ordenadas por prioridad y tiempo restante"""
    pendientes = [u for u in urgencias_diferidas if u.estado == EstadoUrgencia.PENDIENTE]
    return sorted(pendientes, key=lambda x: (x.prioridad, x.horas_restantes))

def programar_urgencia(urgencia_id: str, fecha: date, quirofano: int):
    """Marca una urgencia como programada"""
    for u in urgencias_diferidas:
        if u.id == urgencia_id:
            u.estado = EstadoUrgencia.PROGRAMADA
            u.fecha_programada = fecha
            u.quirofano_asignado = quirofano
            return True
    return False


def programar_urgencias_automaticamente():
    """
    Programa automáticamente las urgencias pendientes en los huecos reservados.
    
    Returns:
        Tuple: (resumen_md, urgencias_programadas, urgencias_sin_hueco)
    """
    global urgencias_diferidas, ultimo_resultado_sesiones
    
    resultado = ultimo_resultado_sesiones or (programador.ultimo_resultado if programador else None)
    
    if not resultado or not resultado.programa:
        return "⚠️ No hay programa optimizado. Ejecuta primero una optimización.", [], []
    
    # Obtener reservas ML
    reservas_por_esp = {}
    if PREDICTOR_DISPONIBLE and predictor_urgencias is not None:
        try:
            reservas_por_esp = predictor_urgencias.obtener_reservas_por_especialidad()
        except:
            pass
    
    # Obtener urgencias pendientes ordenadas por criticidad
    urgencias_pendientes = [u for u in urgencias_diferidas if u.estado == EstadoUrgencia.PENDIENTE]
    urgencias_pendientes.sort(key=lambda u: (u.prioridad, u.horas_restantes))
    
    if not urgencias_pendientes:
        return "✅ No hay urgencias pendientes para programar.", [], []
    
    # Estructura para trackear huecos disponibles por día/quirófano/turno
    # Calcular huecos de reserva disponibles
    huecos_reserva = {}  # {(fecha, q, turno): {'esp': str, 'minutos_disponibles': int, 'hora_inicio': int}}
    
    fechas = sorted(resultado.programa.programas_diarios.keys())
    
    for fecha in fechas:
        if fecha.weekday() >= 5:  # Solo días laborables
            continue
            
        dia_nombre = DIAS_SEMANA[fecha.weekday()]
        prog_dia = resultado.programa.programas_diarios[fecha]
        
        for q in range(1, NUM_QUIROFANOS + 1):
            for turno in TURNOS:
                esp = configuracion_sesiones[q][dia_nombre][turno]
                if esp in ['LIBRE', 'CERRADO']:
                    continue
                
                # Calcular reserva para esta sesión
                if turno == 'Mañana':
                    h_ini, h_fin = HORARIO_MANANA_INICIO, HORARIO_MANANA_FIN
                else:
                    h_ini, h_fin = HORARIO_TARDE_INICIO, HORARIO_TARDE_FIN
                
                duracion_turno = h_fin - h_ini
                
                # Obtener % de reserva
                reserva_pct = reservas_por_esp.get(esp, {}).get('pct_reserva', 15.0)
                if fecha.weekday() in [0, 1]:
                    reserva_pct *= 1.15
                elif fecha.weekday() == 4:
                    reserva_pct *= 0.85
                
                reserva_min = int(duracion_turno * reserva_pct / 100)
                
                if reserva_min > 0:
                    # Verificar si ya hay urgencias asignadas a este hueco
                    urgencias_ya_asignadas = [u for u in urgencias_diferidas 
                                               if u.estado == EstadoUrgencia.PROGRAMADA 
                                               and u.fecha_programada == fecha 
                                               and u.quirofano_asignado == q]
                    
                    tiempo_ocupado = sum(u.duracion_estimada_min for u in urgencias_ya_asignadas)
                    tiempo_disponible = max(0, reserva_min - tiempo_ocupado)
                    
                    if tiempo_disponible > 0:
                        huecos_reserva[(fecha, q, turno)] = {
                            'esp': esp,
                            'minutos_disponibles': tiempo_disponible,
                            'minutos_totales': reserva_min,
                            'hora_inicio': h_fin - reserva_min + tiempo_ocupado
                        }
    
    # Intentar asignar cada urgencia
    programadas = []
    sin_hueco = []
    
    for urgencia in urgencias_pendientes:
        asignada = False
        motivo_fallo = ""
        sugerencia = ""
        
        # Buscar hueco compatible
        huecos_compatibles = []
        for (fecha, q, turno), hueco in huecos_reserva.items():
            if hueco['esp'] == urgencia.especialidad and hueco['minutos_disponibles'] >= urgencia.duracion_estimada_min:
                huecos_compatibles.append((fecha, q, turno, hueco))
        
        if huecos_compatibles:
            # Ordenar por fecha (más pronto primero)
            huecos_compatibles.sort(key=lambda x: x[0])
            fecha, q, turno, hueco = huecos_compatibles[0]
            
            # Asignar
            urgencia.estado = EstadoUrgencia.PROGRAMADA
            urgencia.fecha_programada = fecha
            urgencia.quirofano_asignado = q
            
            # Actualizar hueco disponible
            huecos_reserva[(fecha, q, turno)]['minutos_disponibles'] -= urgencia.duracion_estimada_min
            huecos_reserva[(fecha, q, turno)]['hora_inicio'] += urgencia.duracion_estimada_min
            
            hora_str = f"{hueco['hora_inicio']//60:02d}:{hueco['hora_inicio']%60:02d}"
            programadas.append({
                'id': urgencia.id,
                'paciente': urgencia.paciente_nombre,
                'especialidad': urgencia.especialidad,
                'duracion': urgencia.duracion_estimada_min,
                'fecha': fecha,
                'quirofano': q,
                'turno': turno,
                'hora': hora_str,
                'horas_restantes': urgencia.horas_restantes
            })
            asignada = True
        
        if not asignada:
            # Determinar motivo del fallo
            esp_nombre = ESPECIALIDADES_NOMBRES.get(urgencia.especialidad, urgencia.especialidad)
            
            # Verificar si hay sesiones de esa especialidad
            sesiones_esp = [(f, q, t, h) for (f, q, t), h in huecos_reserva.items() if h['esp'] == urgencia.especialidad]
            
            if not sesiones_esp:
                # No hay ninguna sesión de esa especialidad con reserva
                todas_sesiones = any(
                    configuracion_sesiones[q][d][t] == urgencia.especialidad 
                    for q in range(1, NUM_QUIROFANOS + 1) 
                    for d in DIAS_SEMANA 
                    for t in TURNOS
                )
                if todas_sesiones:
                    motivo_fallo = "RESERVA_AGOTADA"
                    sugerencia = f"Todas las reservas de {esp_nombre} están ocupadas. Usar tiempo libre o abrir sesión extra."
                else:
                    motivo_fallo = "SIN_SESION"
                    sugerencia = f"No hay sesiones de {esp_nombre} configuradas. Añadir sesión o derivar a otro centro."
            else:
                # Hay sesiones pero no cabe
                max_disponible = max(h['minutos_disponibles'] for _, _, _, h in sesiones_esp)
                motivo_fallo = "DURACION_EXCEDE"
                sugerencia = f"La urgencia requiere {urgencia.duracion_estimada_min} min pero el mayor hueco disponible es {max_disponible} min."
            
            sin_hueco.append({
                'id': urgencia.id,
                'paciente': urgencia.paciente_nombre,
                'especialidad': urgencia.especialidad,
                'duracion': urgencia.duracion_estimada_min,
                'prioridad': urgencia.prioridad,
                'horas_restantes': urgencia.horas_restantes,
                'es_critica': urgencia.es_critica,
                'motivo': motivo_fallo,
                'sugerencia': sugerencia
            })
    
    # Generar resumen
    resumen = f"""
## 📋 Resultado de Programación de Urgencias

**Total procesadas:** {len(urgencias_pendientes)} | **Programadas:** {len(programadas)} | **Sin hueco:** {len(sin_hueco)}

"""
    
    if programadas:
        resumen += """### ✅ Urgencias Programadas

| ID | Paciente | Especialidad | Duración | Asignación | ⏱️ Restante |
|----|----------|--------------|----------|------------|-------------|
"""
        for p in programadas:
            esp_nombre = ESPECIALIDADES_NOMBRES.get(p['especialidad'], p['especialidad'])
            fecha_str = p['fecha'].strftime('%a %d/%m')
            resumen += f"| {p['id']} | {p['paciente'][:20]} | {esp_nombre[:15]} | {p['duracion']} min | {fecha_str} Q{p['quirofano']} {p['hora']} | {p['horas_restantes']:.0f}h |\n"
    
    if sin_hueco:
        resumen += """
### ⚠️ Urgencias SIN HUECO DISPONIBLE

| ID | Paciente | Especialidad | Duración | Problema | ⏱️ Restante |
|----|----------|--------------|----------|----------|-------------|
"""
        for s in sin_hueco:
            esp_nombre = ESPECIALIDADES_NOMBRES.get(s['especialidad'], s['especialidad'])
            icono = "🚨" if s['es_critica'] else "⚠️"
            motivo_texto = {
                'SIN_SESION': 'Sin sesión',
                'RESERVA_AGOTADA': 'Reservas llenas',
                'DURACION_EXCEDE': 'No cabe'
            }.get(s['motivo'], s['motivo'])
            resumen += f"| {icono} {s['id']} | {s['paciente'][:20]} | {esp_nombre[:15]} | {s['duracion']} min | {motivo_texto} | {s['horas_restantes']:.0f}h |\n"
        
        resumen += "\n### 💡 Sugerencias\n\n"
        for s in sin_hueco:
            resumen += f"- **{s['id']}**: {s['sugerencia']}\n"
    
    # Alerta crítica si hay urgencias críticas sin asignar
    criticas_sin_hueco = [s for s in sin_hueco if s['es_critica']]
    if criticas_sin_hueco:
        resumen = f"""
## 🚨 ALERTA CRÍTICA

**{len(criticas_sin_hueco)} urgencias con <12h restantes NO TIENEN HUECO:**

""" + "\n".join([f"- **{s['id']}** ({s['paciente']}) - {ESPECIALIDADES_NOMBRES.get(s['especialidad'], s['especialidad'])} - ⏱️ {s['horas_restantes']:.1f}h restantes" for s in criticas_sin_hueco]) + "\n\n**Acción requerida:** Abrir quirófano extra, usar tiempo libre, o derivar.\n\n---\n" + resumen
    
    return resumen, programadas, sin_hueco


# Generar algunas urgencias de ejemplo
def generar_urgencias_ejemplo():
    """Genera urgencias de ejemplo para demostración"""
    global urgencias_diferidas
    urgencias_diferidas = []
    
    ejemplos = [
        ("Juan García López", 67, "CIRUGIA_GENERAL", "Apendicitis aguda", "Apendicectomía lap.", 60, 24, 1),
        ("María Fernández Ruiz", 45, "CIRUGIA_DIGESTIVA", "Colecistitis aguda", "Colecistectomía lap.", 90, 48, 2),
        ("Pedro Martínez Sanz", 72, "CIRUGIA_VASCULAR", "Isquemia aguda MMII", "Embolectomía", 120, 24, 1),
        ("Ana López García", 58, "GINECOLOGIA", "Torsión ovárica", "Ooforectomía lap.", 75, 24, 1),
        ("Carlos Ruiz Pérez", 55, "UROLOGIA", "Retención urinaria", "RTU próstata", 60, 72, 3),
        ("Laura Sánchez Díaz", 38, "CIRUGIA_GENERAL", "Hernia incarcerada", "Hernioplastia", 90, 48, 2),
    ]
    
    for paciente, edad, esp, diag, proc, dur, horas, prio in ejemplos:
        agregar_urgencia(paciente, edad, esp, diag, proc, dur, horas, prio)

generar_urgencias_ejemplo()


def generar_resumen_configuracion():
    conteo = {}
    for q in range(1, NUM_QUIROFANOS + 1):
        for dia in DIAS_SEMANA:
            for turno in TURNOS:
                esp = configuracion_sesiones[q][dia][turno]
                if esp not in ['LIBRE', 'CERRADO']:
                    if esp not in conteo:
                        conteo[esp] = {'M': 0, 'T': 0}
                    if turno == 'Mañana':
                        conteo[esp]['M'] += 1
                    else:
                        conteo[esp]['T'] += 1
    
    texto = "### 📊 Resumen de Sesiones\n\n| Especialidad | Mañanas | Tardes | Total | Horas/sem |\n|--------------|---------|--------|-------|-----------|"
    for esp, c in sorted(conteo.items(), key=lambda x: x[1]['M'] + x[1]['T'], reverse=True):
        total = c['M'] + c['T']
        horas = c['M'] * 7 + c['T'] * 5
        nombre = ESPECIALIDADES_NOMBRES.get(esp, esp)
        texto += f"\n| {nombre} | {c['M']} | {c['T']} | {total} | {horas}h |"
    return texto


def generar_matriz_visual():
    datos = []
    for q in range(1, NUM_QUIROFANOS + 1):
        fila = {'Q': f'Q{q}'}
        for dia in DIAS_SEMANA:
            esp_m = configuracion_sesiones[q][dia]['Mañana']
            esp_t = configuracion_sesiones[q][dia]['Tarde']
            fila[f'{dia[:3]}M'] = ESPECIALIDADES_NOMBRES.get(esp_m, esp_m)[:8]
            fila[f'{dia[:3]}T'] = ESPECIALIDADES_NOMBRES.get(esp_t, esp_t)[:8]
        datos.append(fila)
    return pd.DataFrame(datos)


def generar_grafico_sesiones():
    matriz = np.zeros((NUM_QUIROFANOS, len(DIAS_SEMANA) * 2))
    esp_to_num = {esp: i+1 for i, esp in enumerate(LISTA_ESPECIALIDADES)}
    
    for q in range(NUM_QUIROFANOS):
        for d, dia in enumerate(DIAS_SEMANA):
            esp_m = configuracion_sesiones[q+1][dia]['Mañana']
            esp_t = configuracion_sesiones[q+1][dia]['Tarde']
            matriz[q, d*2] = esp_to_num.get(esp_m, 0)
            matriz[q, d*2+1] = esp_to_num.get(esp_t, 0)
    
    cols = [f'{d[:3]}{t}' for d in DIAS_SEMANA for t in ['M', 'T']]
    fig = px.imshow(matriz, labels={'x': 'Día/Turno', 'y': 'Quirófano', 'color': 'Esp'},
                   x=cols, y=[f'Q{i}' for i in range(1, NUM_QUIROFANOS+1)],
                   color_continuous_scale='Viridis', aspect='auto')
    fig.update_layout(height=300, margin=dict(l=50, r=50, t=30, b=30))
    return fig


def guardar_configuracion(*args):
    idx = 0
    for q in range(1, NUM_QUIROFANOS + 1):
        for dia in DIAS_SEMANA:
            for turno in TURNOS:
                configuracion_sesiones[q][dia][turno] = args[idx]
                idx += 1
    return ("✅ **Configuración guardada.**", generar_resumen_configuracion(), 
            generar_matriz_visual(), generar_grafico_sesiones())


def resetear_configuracion():
    inicializar_configuracion_default()
    valores = []
    for q in range(1, NUM_QUIROFANOS + 1):
        for dia in DIAS_SEMANA:
            for turno in TURNOS:
                valores.append(configuracion_sesiones[q][dia][turno])
    return ["✅ **Configuración reseteada.**", generar_resumen_configuracion(), 
            generar_matriz_visual(), generar_grafico_sesiones()] + valores


# =============================================================================
# SISTEMA DE RESTRICCIONES PERSONALIZADAS
# =============================================================================

# Lista global de restricciones activas
restricciones_usuario = []

# Catálogo de tipos de restricciones disponibles
CATALOGO_RESTRICCIONES = {
    'cerrar_esp_dia': {
        'nombre': 'Cerrar especialidad un día',
        'descripcion': 'Una especialidad no opera un día específico',
        'plantilla': '{especialidad} no opera el {dia} ({turno})',
        'parametros': ['especialidad', 'dia', 'turno'],
    },
    'cerrar_quirofano': {
        'nombre': 'Cerrar quirófano',
        'descripcion': 'Un quirófano no está disponible',
        'plantilla': 'Q{quirofano} cerrado el {dia} ({turno})',
        'parametros': ['quirofano', 'dia', 'turno'],
    },
    'esp_solo_manana': {
        'nombre': 'Especialidad solo mañanas',
        'descripcion': 'Una especialidad solo puede operar en turno de mañana',
        'plantilla': '{especialidad} solo opera en turno de mañana',
        'parametros': ['especialidad'],
    },
    'esp_solo_tarde': {
        'nombre': 'Especialidad solo tardes',
        'descripcion': 'Una especialidad solo puede operar en turno de tarde',
        'plantilla': '{especialidad} solo opera en turno de tarde',
        'parametros': ['especialidad'],
    },
    'max_sesiones_esp': {
        'nombre': 'Máximo sesiones por especialidad',
        'descripcion': 'Limitar el número de sesiones semanales de una especialidad',
        'plantilla': '{especialidad} máximo {cantidad} sesiones/semana',
        'parametros': ['especialidad', 'cantidad'],
    },
    'min_sesiones_esp': {
        'nombre': 'Mínimo sesiones por especialidad',
        'descripcion': 'Garantizar un mínimo de sesiones semanales',
        'plantilla': '{especialidad} mínimo {cantidad} sesiones/semana',
        'parametros': ['especialidad', 'cantidad'],
    },
    'esp_en_quirofano': {
        'nombre': 'Especialidad en quirófano fijo',
        'descripcion': 'Una especialidad solo puede operar en un quirófano específico',
        'plantilla': '{especialidad} solo en Q{quirofano}',
        'parametros': ['especialidad', 'quirofano'],
    },
    'quirofano_para_esp': {
        'nombre': 'Reservar quirófano para especialidad',
        'descripcion': 'Un quirófano se reserva exclusivamente para una especialidad',
        'plantilla': 'Q{quirofano} reservado para {especialidad}',
        'parametros': ['quirofano', 'especialidad'],
    },
    'cerrar_dia_completo': {
        'nombre': 'Cerrar día completo',
        'descripcion': 'No operar ningún quirófano un día específico',
        'plantilla': 'No operar el {dia} (todos los quirófanos)',
        'parametros': ['dia'],
    },
    'cerrar_turno_dia': {
        'nombre': 'Cerrar turno en un día',
        'descripcion': 'Cerrar todos los quirófanos en un turno específico de un día',
        'plantilla': 'No operar en turno de {turno} el {dia}',
        'parametros': ['turno', 'dia'],
    },
}

# Opciones para los dropdowns
OPCIONES_TURNO = ['Mañana', 'Tarde', 'Ambos']
OPCIONES_QUIROFANO = [str(i) for i in range(1, NUM_QUIROFANOS + 1)]
OPCIONES_CANTIDAD = [str(i) for i in range(1, 16)]
ESPECIALIDADES_SIN_LIBRE = [e for e in LISTA_ESPECIALIDADES if e not in ['LIBRE', 'CERRADO']]


def generar_texto_restriccion(tipo, params):
    """Genera el texto descriptivo de una restricción"""
    if tipo not in CATALOGO_RESTRICCIONES:
        return "Restricción desconocida"
    
    plantilla = CATALOGO_RESTRICCIONES[tipo]['plantilla']
    
    # Reemplazar parámetros
    texto = plantilla
    for key, value in params.items():
        if key == 'especialidad':
            valor_mostrar = ESPECIALIDADES_NOMBRES.get(value, value)
        elif key == 'quirofano':
            valor_mostrar = value
        else:
            valor_mostrar = value
        texto = texto.replace('{' + key + '}', str(valor_mostrar))
    
    return texto


def añadir_restriccion(tipo, especialidad, dia, turno, quirofano, cantidad):
    """Añade una restricción a la lista"""
    global restricciones_usuario
    
    if not tipo:
        return actualizar_lista_restricciones(), "⚠️ Selecciona un tipo de restricción"
    
    # Construir parámetros según el tipo
    params = {}
    tipo_info = CATALOGO_RESTRICCIONES.get(tipo, {})
    parametros_necesarios = tipo_info.get('parametros', [])
    
    for p in parametros_necesarios:
        if p == 'especialidad':
            if not especialidad:
                return actualizar_lista_restricciones(), "⚠️ Selecciona una especialidad"
            params['especialidad'] = especialidad
        elif p == 'dia':
            if not dia:
                return actualizar_lista_restricciones(), "⚠️ Selecciona un día"
            params['dia'] = dia
        elif p == 'turno':
            if not turno:
                return actualizar_lista_restricciones(), "⚠️ Selecciona un turno"
            params['turno'] = turno
        elif p == 'quirofano':
            if not quirofano:
                return actualizar_lista_restricciones(), "⚠️ Selecciona un quirófano"
            params['quirofano'] = quirofano
        elif p == 'cantidad':
            if not cantidad:
                return actualizar_lista_restricciones(), "⚠️ Selecciona una cantidad"
            params['cantidad'] = cantidad
    
    # Crear restricción
    restriccion = {
        'id': len(restricciones_usuario) + 1,
        'tipo': tipo,
        'params': params,
        'texto': generar_texto_restriccion(tipo, params)
    }
    
    restricciones_usuario.append(restriccion)
    
    return actualizar_lista_restricciones(), f"✅ Restricción añadida: {restriccion['texto']}"


def eliminar_restriccion(indice):
    """Elimina una restricción por su índice"""
    global restricciones_usuario
    
    try:
        idx = int(indice) - 1
        if 0 <= idx < len(restricciones_usuario):
            eliminada = restricciones_usuario.pop(idx)
            # Renumerar
            for i, r in enumerate(restricciones_usuario):
                r['id'] = i + 1
            return actualizar_lista_restricciones(), f"✅ Eliminada: {eliminada['texto']}"
    except:
        pass
    
    return actualizar_lista_restricciones(), "⚠️ Índice no válido"


def limpiar_restricciones():
    """Elimina todas las restricciones"""
    global restricciones_usuario
    restricciones_usuario = []
    return actualizar_lista_restricciones(), "✅ Todas las restricciones eliminadas"


def actualizar_lista_restricciones():
    """Genera el markdown con la lista de restricciones activas"""
    if not restricciones_usuario:
        return "📋 **No hay restricciones activas**\n\nLa configuración óptima se calculará sin restricciones adicionales."
    
    texto = f"📋 **{len(restricciones_usuario)} restricción(es) activa(s)**\n\n"
    for r in restricciones_usuario:
        texto += f"{r['id']}. 🚫 {r['texto']}\n"
    
    return texto


def aplicar_restricciones_a_config(config_base):
    """Aplica las restricciones del usuario a una configuración de sesiones"""
    import copy
    config = copy.deepcopy(config_base)
    
    for restriccion in restricciones_usuario:
        tipo = restriccion['tipo']
        params = restriccion['params']
        
        if tipo == 'cerrar_esp_dia':
            # Cerrar especialidad un día específico
            esp = params['especialidad']
            dia = params['dia']
            turno = params['turno']
            
            for q in range(1, NUM_QUIROFANOS + 1):
                if turno in ['Mañana', 'Ambos']:
                    if config[q][dia]['Mañana'] == esp:
                        config[q][dia]['Mañana'] = 'CERRADO'
                if turno in ['Tarde', 'Ambos']:
                    if config[q][dia]['Tarde'] == esp:
                        config[q][dia]['Tarde'] = 'CERRADO'
        
        elif tipo == 'cerrar_quirofano':
            # Cerrar quirófano específico
            q = int(params['quirofano'])
            dia = params['dia']
            turno = params['turno']
            
            if turno in ['Mañana', 'Ambos']:
                config[q][dia]['Mañana'] = 'CERRADO'
            if turno in ['Tarde', 'Ambos']:
                config[q][dia]['Tarde'] = 'CERRADO'
        
        elif tipo == 'esp_solo_manana':
            # Especialidad solo puede operar mañanas
            esp = params['especialidad']
            for q in range(1, NUM_QUIROFANOS + 1):
                for dia in DIAS_SEMANA:
                    if config[q][dia]['Tarde'] == esp:
                        config[q][dia]['Tarde'] = 'LIBRE'
        
        elif tipo == 'esp_solo_tarde':
            # Especialidad solo puede operar tardes
            esp = params['especialidad']
            for q in range(1, NUM_QUIROFANOS + 1):
                for dia in DIAS_SEMANA:
                    if config[q][dia]['Mañana'] == esp:
                        config[q][dia]['Mañana'] = 'LIBRE'
        
        elif tipo == 'cerrar_dia_completo':
            # Cerrar todos los quirófanos un día
            dia = params['dia']
            for q in range(1, NUM_QUIROFANOS + 1):
                config[q][dia]['Mañana'] = 'CERRADO'
                config[q][dia]['Tarde'] = 'CERRADO'
        
        elif tipo == 'cerrar_turno_dia':
            # Cerrar turno completo un día
            dia = params['dia']
            turno = params['turno']
            for q in range(1, NUM_QUIROFANOS + 1):
                config[q][dia][turno] = 'CERRADO'
        
        elif tipo == 'esp_en_quirofano':
            # Especialidad solo en un quirófano específico
            esp = params['especialidad']
            q_permitido = int(params['quirofano'])
            for q in range(1, NUM_QUIROFANOS + 1):
                if q != q_permitido:
                    for dia in DIAS_SEMANA:
                        if config[q][dia]['Mañana'] == esp:
                            config[q][dia]['Mañana'] = 'LIBRE'
                        if config[q][dia]['Tarde'] == esp:
                            config[q][dia]['Tarde'] = 'LIBRE'
        
        elif tipo == 'quirofano_para_esp':
            # Quirófano reservado para una especialidad
            q_reservado = int(params['quirofano'])
            esp = params['especialidad']
            for dia in DIAS_SEMANA:
                for turno in TURNOS:
                    if config[q_reservado][dia][turno] not in ['CERRADO', 'LIBRE']:
                        config[q_reservado][dia][turno] = esp
    
    # Para restricciones de máximo/mínimo sesiones, se aplican durante el cálculo óptimo
    return config


def obtener_limites_sesiones():
    """Obtiene los límites de sesiones por especialidad de las restricciones"""
    limites = {}
    
    for restriccion in restricciones_usuario:
        tipo = restriccion['tipo']
        params = restriccion['params']
        
        if tipo == 'max_sesiones_esp':
            esp = params['especialidad']
            cantidad = int(params['cantidad'])
            if esp not in limites:
                limites[esp] = {'min': 0, 'max': float('inf')}
            limites[esp]['max'] = min(limites[esp]['max'], cantidad)
        
        elif tipo == 'min_sesiones_esp':
            esp = params['especialidad']
            cantidad = int(params['cantidad'])
            if esp not in limites:
                limites[esp] = {'min': 0, 'max': float('inf')}
            limites[esp]['min'] = max(limites[esp]['min'], cantidad)
    
    return limites


def get_parametros_visibles(tipo_restriccion):
    """Determina qué parámetros mostrar según el tipo de restricción"""
    if not tipo_restriccion or tipo_restriccion not in CATALOGO_RESTRICCIONES:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    params = CATALOGO_RESTRICCIONES[tipo_restriccion]['parametros']
    
    return (
        gr.update(visible='especialidad' in params),
        gr.update(visible='dia' in params),
        gr.update(visible='turno' in params),
        gr.update(visible='quirofano' in params),
        gr.update(visible='cantidad' in params)
    )
    conteo = {}
    for q in range(1, NUM_QUIROFANOS + 1):
        for dia in DIAS_SEMANA:
            for turno in TURNOS:
                esp = configuracion_sesiones[q][dia][turno]
                if esp not in ['LIBRE', 'CERRADO']:
                    if esp not in conteo:
                        conteo[esp] = {'Mañana': 0, 'Tarde': 0, 'Total': 0}
                    conteo[esp][turno] += 1
                    conteo[esp]['Total'] += 1
    
    resumen = "## 📊 Resumen de Configuración\n\n"
    resumen += "| Especialidad | Mañanas | Tardes | Total |\n|--------------|---------|--------|-------|\n"
    
    for esp, counts in sorted(conteo.items(), key=lambda x: -x[1]['Total']):
        nombre = ESPECIALIDADES_NOMBRES.get(esp, esp)
        resumen += f"| {nombre} | {counts['Mañana']} | {counts['Tarde']} | {counts['Total']} |\n"
    
    total_m = sum(1 for q in range(1, NUM_QUIROFANOS + 1) for dia in DIAS_SEMANA 
                  if configuracion_sesiones[q][dia]['Mañana'] not in ['LIBRE', 'CERRADO'])
    total_t = sum(1 for q in range(1, NUM_QUIROFANOS + 1) for dia in DIAS_SEMANA 
                  if configuracion_sesiones[q][dia]['Tarde'] not in ['LIBRE', 'CERRADO'])
    
    resumen += f"\n**Total semanal:** {total_m} mañanas ({total_m*7}h) + {total_t} tardes ({total_t*5}h) = **{total_m*7 + total_t*5} horas**"
    return resumen


def generar_matriz_visual():
    datos = []
    for q in range(1, NUM_QUIROFANOS + 1):
        fila = {'Q': f'Q{q}'}
        for dia in DIAS_SEMANA:
            esp_m = configuracion_sesiones[q][dia]['Mañana']
            esp_t = configuracion_sesiones[q][dia]['Tarde']
            fila[f'{dia[:3]}M'] = ESPECIALIDADES_NOMBRES.get(esp_m, esp_m)[:8]
            fila[f'{dia[:3]}T'] = ESPECIALIDADES_NOMBRES.get(esp_t, esp_t)[:8]
        datos.append(fila)
    return pd.DataFrame(datos)


def generar_grafico_sesiones():
    esp_to_idx = {esp: i for i, esp in enumerate(LISTA_ESPECIALIDADES)}
    z_data, y_labels, hover_text = [], [], []
    
    for q in range(1, NUM_QUIROFANOS + 1):
        y_labels.append(f'Q{q}')
        fila_z, fila_hover = [], []
        for dia in DIAS_SEMANA:
            for turno in TURNOS:
                esp = configuracion_sesiones[q][dia][turno]
                fila_z.append(esp_to_idx.get(esp, 0))
                fila_hover.append(f'Q{q} - {dia} {turno}<br><b>{ESPECIALIDADES_NOMBRES.get(esp, esp)}</b>')
        z_data.append(fila_z)
        hover_text.append(fila_hover)
    
    x_labels = [f'{dia[:3]}_{turno[0]}' for dia in DIAS_SEMANA for turno in TURNOS]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data, x=x_labels, y=y_labels, hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        colorscale=[[i/15, c] for i, c in enumerate([
            '#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#e91e63',
            '#00bcd4', '#ff9800', '#ff5722', '#795548', '#9c27b0', '#607d8b',
            '#4caf50', '#03a9f4', '#ecf0f1', '#bdc3c7'])],
        showscale=False, xgap=2, ygap=2
    ))
    fig.update_layout(title='Mapa de Sesiones (M=Mañana, T=Tarde)', height=300, 
                      yaxis=dict(autorange='reversed'))
    return fig


def guardar_configuracion(*args):
    global configuracion_sesiones
    idx = 0
    for q in range(1, NUM_QUIROFANOS + 1):
        for dia in DIAS_SEMANA:
            for turno in TURNOS:
                if idx < len(args):
                    configuracion_sesiones[q][dia][turno] = args[idx]
                idx += 1
    return ("✅ **Configuración guardada.**", generar_resumen_configuracion(), 
            generar_matriz_visual(), generar_grafico_sesiones())


def resetear_configuracion():
    inicializar_configuracion_default()
    valores = []
    for q in range(1, NUM_QUIROFANOS + 1):
        for dia in DIAS_SEMANA:
            for turno in TURNOS:
                valores.append(configuracion_sesiones[q][dia][turno])
    return ["✅ **Configuración reseteada.**", generar_resumen_configuracion(), 
            generar_matriz_visual(), generar_grafico_sesiones()] + valores


# =============================================================================
# OPTIMIZADOR CON SESIONES
# =============================================================================

class ResultadoOpt:
    def __init__(self):
        self.programa = None
        self.score_total = 0.0
        self.score_clinico = 0.0
        self.score_eficiencia = 0.0
        self.cirugias_programadas = 0
        self.cirugias_no_programadas = 0
        self.robustez_score = 0.0
        self.tiempo_ejecucion_seg = 0.0
        self.reservas_aplicadas = {}  # info de reservas usadas
        self.capacidad_efectiva_total = 0
        self.capacidad_reservada_total = 0
        self.predictor_ml_activo = False  # True si se usó ML
        self.motivos_no_programacion = {}  # Razones de no programación
        self.metodo_usado = 'heuristico'  # Método de optimización usado
        self.comparativa_metodos = None  # Para modo auto


# =============================================================================
# MÉTODOS DE OPTIMIZACIÓN
# =============================================================================

def _optimizar_heuristico(solicitudes_validas, estado_slots, cirujanos, fecha_inicio):
    """
    Método heurístico: First Fit Decreasing por prioridad clínica.
    Rápido, buena solución inicial.
    """
    solicitudes_ord = sorted(solicitudes_validas, key=lambda x: x.score_clinico, reverse=True)
    ids_prog = set()
    asignaciones = {}
    
    # Clonar estado_slots para no modificar original
    slots = {k: dict(v) for k, v in estado_slots.items()}
    
    for sol in solicitudes_ord:
        esp = sol.tipo_intervencion.especialidad.name
        dur = sol.duracion_esperada()
        
        mejor = None
        for (dia, q, turno), slot in slots.items():
            if slot['esp'] != esp:
                continue
            if slot['hora_disp'] + dur + 30 > slot['hora_fin'] + 30:
                continue
            
            # Priorizar días cercanos y mañanas
            score = 100 - (dia - fecha_inicio).days * 5 + (10 if turno == 'Mañana' else 0)
            disponible = slot['hora_fin'] - slot['hora_disp']
            if disponible > dur * 1.5:
                score += 5
            
            if mejor is None or score > mejor[0]:
                mejor = (score, dia, q, turno, slot['hora_disp'])
        
        if mejor:
            _, dia, q, turno, hora = mejor
            
            # Asignar cirujano
            cirujano = sol.cirujano_asignado
            if not cirujano:
                cirs = [c for c in cirujanos if c.especialidad_principal.name == esp]
                cirujano = random.choice(cirs) if cirs else None
            
            asignaciones[sol.id] = {
                'dia': dia, 'quirofano': q, 'turno': turno,
                'hora': hora, 'duracion': dur, 'cirujano': cirujano
            }
            slots[(dia, q, turno)]['hora_disp'] = hora + dur + 30
            ids_prog.add(sol.id)
    
    return ids_prog, asignaciones


def _optimizar_genetico(solicitudes_validas, estado_slots, cirujanos, fecha_inicio):
    """
    Método genético: Usa DEAP para evolución de soluciones.
    Mejor calidad pero más lento.
    """
    if not DEAP_AVAILABLE:
        print("⚠️ DEAP no disponible, usando heurístico")
        return _optimizar_heuristico(solicitudes_validas, estado_slots, cirujanos, fecha_inicio)
    
    try:
        from deap import base, creator, tools, algorithms
        
        # Configuración DEAP
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        solicitudes_ord = sorted(solicitudes_validas, key=lambda x: x.score_clinico, reverse=True)
        n_solicitudes = min(len(solicitudes_ord), 200)  # Limitar para velocidad
        solicitudes_subset = solicitudes_ord[:n_solicitudes]
        
        # Crear lista de slots válidos por especialidad
        slots_por_esp = defaultdict(list)
        for (dia, q, turno), slot in estado_slots.items():
            slots_por_esp[slot['esp']].append((dia, q, turno, slot))
        
        def evaluar(individuo):
            """Evalúa una solución (permutación de prioridades)"""
            slots_temp = {k: dict(v) for k, v in estado_slots.items()}
            programadas = 0
            onco_prog = 0
            fp_prog = 0
            
            # Reordenar solicitudes según individuo
            orden = sorted(range(len(solicitudes_subset)), key=lambda i: individuo[i], reverse=True)
            
            for idx in orden:
                sol = solicitudes_subset[idx]
                esp = sol.tipo_intervencion.especialidad.name
                dur = sol.duracion_esperada()
                
                for (dia, q, turno), slot in slots_temp.items():
                    if slot['esp'] != esp:
                        continue
                    if slot['hora_disp'] + dur + 30 <= slot['hora_fin'] + 30:
                        slots_temp[(dia, q, turno)]['hora_disp'] += dur + 30
                        programadas += 1
                        if 'ONCOLOGICO' in sol.prioridad.name:
                            onco_prog += 1
                        if sol.esta_fuera_plazo:
                            fp_prog += 1
                        break
            
            # Fitness: priorizar oncológicos y fuera de plazo
            onco_tot = sum(1 for s in solicitudes_subset if 'ONCOLOGICO' in s.prioridad.name)
            fp_tot = sum(1 for s in solicitudes_subset if s.esta_fuera_plazo)
            
            fitness = (0.5 * (onco_prog / max(1, onco_tot)) + 
                      0.3 * (fp_prog / max(1, fp_tot)) + 
                      0.2 * (programadas / len(solicitudes_subset)))
            return (fitness,)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_solicitudes)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluar)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Evolución
        pop = toolbox.population(n=50)
        
        # Evaluación inicial
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Evolución por generaciones
        for gen in range(30):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.2)
            fitnesses = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, k=len(pop))
        
        # Mejor individuo
        mejor = tools.selBest(pop, k=1)[0]
        
        # Reconstruir solución con el mejor orden
        orden = sorted(range(len(solicitudes_subset)), key=lambda i: mejor[i], reverse=True)
        
        ids_prog = set()
        asignaciones = {}
        slots = {k: dict(v) for k, v in estado_slots.items()}
        
        for idx in orden:
            sol = solicitudes_subset[idx]
            esp = sol.tipo_intervencion.especialidad.name
            dur = sol.duracion_esperada()
            
            mejor_slot = None
            for (dia, q, turno), slot in slots.items():
                if slot['esp'] != esp:
                    continue
                if slot['hora_disp'] + dur + 30 > slot['hora_fin'] + 30:
                    continue
                score = 100 - (dia - fecha_inicio).days * 5 + (10 if turno == 'Mañana' else 0)
                if mejor_slot is None or score > mejor_slot[0]:
                    mejor_slot = (score, dia, q, turno, slot['hora_disp'])
            
            if mejor_slot:
                _, dia, q, turno, hora = mejor_slot
                cirujano = sol.cirujano_asignado
                if not cirujano:
                    cirs = [c for c in cirujanos if c.especialidad_principal.name == esp]
                    cirujano = random.choice(cirs) if cirs else None
                
                asignaciones[sol.id] = {
                    'dia': dia, 'quirofano': q, 'turno': turno,
                    'hora': hora, 'duracion': dur, 'cirujano': cirujano
                }
                slots[(dia, q, turno)]['hora_disp'] = hora + dur + 30
                ids_prog.add(sol.id)
        
        # Añadir resto de solicitudes con heurístico
        for sol in solicitudes_validas:
            if sol.id in ids_prog:
                continue
            esp = sol.tipo_intervencion.especialidad.name
            dur = sol.duracion_esperada()
            
            for (dia, q, turno), slot in slots.items():
                if slot['esp'] != esp:
                    continue
                if slot['hora_disp'] + dur + 30 <= slot['hora_fin'] + 30:
                    cirujano = sol.cirujano_asignado
                    if not cirujano:
                        cirs = [c for c in cirujanos if c.especialidad_principal.name == esp]
                        cirujano = random.choice(cirs) if cirs else None
                    
                    asignaciones[sol.id] = {
                        'dia': dia, 'quirofano': q, 'turno': turno,
                        'hora': slot['hora_disp'], 'duracion': dur, 'cirujano': cirujano
                    }
                    slots[(dia, q, turno)]['hora_disp'] += dur + 30
                    ids_prog.add(sol.id)
                    break
        
        return ids_prog, asignaciones
        
    except Exception as e:
        print(f"⚠️ Error en genético: {e}, usando heurístico")
        return _optimizar_heuristico(solicitudes_validas, estado_slots, cirujanos, fecha_inicio)


def _optimizar_milp(solicitudes_validas, estado_slots, cirujanos, fecha_inicio):
    """
    Método MILP: Programación lineal entera mixta con OR-Tools.
    Óptimo global pero puede ser lento para problemas grandes.
    """
    if not ORTOOLS_AVAILABLE:
        print("⚠️ OR-Tools no disponible, usando heurístico")
        return _optimizar_heuristico(solicitudes_validas, estado_slots, cirujanos, fecha_inicio)
    
    try:
        from ortools.sat.python import cp_model
        
        solicitudes_ord = sorted(solicitudes_validas, key=lambda x: x.score_clinico, reverse=True)
        n_solicitudes = min(len(solicitudes_ord), 150)  # Limitar para velocidad
        solicitudes_subset = solicitudes_ord[:n_solicitudes]
        
        model = cp_model.CpModel()
        
        # Variables: x[i,s] = 1 si solicitud i se asigna a slot s
        x = {}
        slots_list = list(estado_slots.keys())
        
        for i, sol in enumerate(solicitudes_subset):
            esp = sol.tipo_intervencion.especialidad.name
            for j, (dia, q, turno) in enumerate(slots_list):
                if estado_slots[(dia, q, turno)]['esp'] == esp:
                    x[i, j] = model.NewBoolVar(f'x_{i}_{j}')
        
        # Restricción: cada solicitud se asigna a máximo un slot
        for i in range(len(solicitudes_subset)):
            asigs = [x[i, j] for j in range(len(slots_list)) if (i, j) in x]
            if asigs:
                model.Add(sum(asigs) <= 1)
        
        # Restricción: capacidad de cada slot
        for j, (dia, q, turno) in enumerate(slots_list):
            slot = estado_slots[(dia, q, turno)]
            capacidad = slot['hora_fin'] - slot['hora_ini']
            
            demanda = []
            for i, sol in enumerate(solicitudes_subset):
                if (i, j) in x:
                    dur = sol.duracion_esperada() + 30
                    demanda.append(x[i, j] * dur)
            
            if demanda:
                model.Add(sum(demanda) <= capacidad)
        
        # Objetivo: maximizar score clínico ponderado
        objetivo = []
        for i, sol in enumerate(solicitudes_subset):
            peso = sol.score_clinico
            if 'ONCOLOGICO' in sol.prioridad.name:
                peso += 50
            if sol.esta_fuera_plazo:
                peso += 30
            
            for j in range(len(slots_list)):
                if (i, j) in x:
                    # Bonus por día cercano
                    dia = slots_list[j][0]
                    bonus_dia = max(0, 10 - (dia - fecha_inicio).days)
                    objetivo.append(x[i, j] * int(peso + bonus_dia))
        
        model.Maximize(sum(objetivo))
        
        # Resolver con límite de tiempo
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            ids_prog = set()
            asignaciones = {}
            
            for i, sol in enumerate(solicitudes_subset):
                for j, (dia, q, turno) in enumerate(slots_list):
                    if (i, j) in x and solver.Value(x[i, j]) == 1:
                        slot = estado_slots[(dia, q, turno)]
                        esp = sol.tipo_intervencion.especialidad.name
                        
                        cirujano = sol.cirujano_asignado
                        if not cirujano:
                            cirs = [c for c in cirujanos if c.especialidad_principal.name == esp]
                            cirujano = random.choice(cirs) if cirs else None
                        
                        asignaciones[sol.id] = {
                            'dia': dia, 'quirofano': q, 'turno': turno,
                            'hora': slot['hora_ini'], 'duracion': sol.duracion_esperada(),
                            'cirujano': cirujano
                        }
                        ids_prog.add(sol.id)
                        break
            
            # Completar con resto de solicitudes usando heurístico
            slots = {k: dict(v) for k, v in estado_slots.items()}
            for sol_id, asig in asignaciones.items():
                key = (asig['dia'], asig['quirofano'], asig['turno'])
                if key in slots:
                    slots[key]['hora_disp'] = max(slots[key]['hora_disp'], asig['hora'] + asig['duracion'] + 30)
            
            for sol in solicitudes_validas:
                if sol.id in ids_prog:
                    continue
                esp = sol.tipo_intervencion.especialidad.name
                dur = sol.duracion_esperada()
                
                for (dia, q, turno), slot in slots.items():
                    if slot['esp'] != esp:
                        continue
                    if slot['hora_disp'] + dur + 30 <= slot['hora_fin'] + 30:
                        cirujano = sol.cirujano_asignado
                        if not cirujano:
                            cirs = [c for c in cirujanos if c.especialidad_principal.name == esp]
                            cirujano = random.choice(cirs) if cirs else None
                        
                        asignaciones[sol.id] = {
                            'dia': dia, 'quirofano': q, 'turno': turno,
                            'hora': slot['hora_disp'], 'duracion': dur, 'cirujano': cirujano
                        }
                        slots[(dia, q, turno)]['hora_disp'] += dur + 30
                        ids_prog.add(sol.id)
                        break
            
            return ids_prog, asignaciones
        else:
            print("⚠️ MILP sin solución factible, usando heurístico")
            return _optimizar_heuristico(solicitudes_validas, estado_slots, cirujanos, fecha_inicio)
            
    except Exception as e:
        print(f"⚠️ Error en MILP: {e}, usando heurístico")
        return _optimizar_heuristico(solicitudes_validas, estado_slots, cirujanos, fecha_inicio)


def _optimizar_auto(solicitudes, cirujanos, fecha_inicio, horizonte_dias,
                    usar_reservas_predictivas, pct_programacion):
    """
    Modo AUTO: Ejecuta los 3 métodos y selecciona el mejor.
    Devuelve el resultado junto con la comparativa.
    """
    print("\n🔄 Modo AUTO: Ejecutando los 3 métodos de optimización...")
    
    resultados = {}
    metodos = ['heuristico']
    
    if DEAP_AVAILABLE:
        metodos.append('genetico')
    if ORTOOLS_AVAILABLE:
        metodos.append('milp')
    
    for metodo in metodos:
        print(f"   → Ejecutando {metodo.upper()}...")
        try:
            r = optimizar_con_sesiones(
                solicitudes, cirujanos, fecha_inicio, horizonte_dias,
                usar_reservas_predictivas, pct_programacion,
                metodo=metodo
            )
            resultados[metodo] = {
                'resultado': r,
                'score_total': r.score_total,
                'score_clinico': r.score_clinico,
                'score_eficiencia': r.score_eficiencia,
                'programadas': r.cirugias_programadas,
                'tiempo': r.tiempo_ejecucion_seg
            }
            print(f"     Score: {r.score_total:.4f} | Programadas: {r.cirugias_programadas} | Tiempo: {r.tiempo_ejecucion_seg:.2f}s")
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    # Seleccionar el mejor por score_total
    mejor_metodo = max(resultados.keys(), key=lambda m: resultados[m]['score_total'])
    mejor_resultado = resultados[mejor_metodo]['resultado']
    
    print(f"\n✅ Mejor método: {mejor_metodo.upper()} (Score: {resultados[mejor_metodo]['score_total']:.4f})")
    
    # Añadir comparativa al resultado
    mejor_resultado.metodo_usado = mejor_metodo
    mejor_resultado.comparativa_metodos = resultados
    
    return mejor_resultado


def optimizar_con_sesiones(solicitudes, cirujanos, fecha_inicio, horizonte_dias=10, 
                           usar_reservas_predictivas=True, pct_programacion=85,
                           metodo='heuristico'):
    """
    Optimiza el programa quirúrgico respetando sesiones y reservas.
    
    Args:
        solicitudes: Lista de solicitudes de cirugía
        cirujanos: Lista de cirujanos disponibles
        fecha_inicio: Fecha de inicio del horizonte
        horizonte_dias: Días a programar
        usar_reservas_predictivas: Si True, aplica reservas ML por especialidad
        pct_programacion: % de capacidad a programar (para dejar margen)
        metodo: 'heuristico', 'genetico', 'milp', o 'auto'
    """
    inicio = time.time()
    
    # Si es "auto", ejecutar los 3 métodos y devolver el mejor
    if metodo == 'auto':
        return _optimizar_auto(solicitudes, cirujanos, fecha_inicio, horizonte_dias,
                               usar_reservas_predictivas, pct_programacion)
    
    fecha_fin = fecha_inicio + timedelta(days=horizonte_dias)
    periodo = ProgramaPeriodo(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    
    dias_habiles = [fecha_inicio + timedelta(days=i) for i in range(horizonte_dias + 5)
                   if (fecha_inicio + timedelta(days=i)).weekday() < 5][:horizonte_dias]
    
    # Calcular reservas por especialidad si está disponible el predictor
    reservas_por_esp = {}
    predictor_activo = False
    
    if usar_reservas_predictivas and PREDICTOR_DISPONIBLE and predictor_urgencias is not None:
        try:
            reservas_por_esp = predictor_urgencias.obtener_reservas_por_especialidad()
            if reservas_por_esp:
                predictor_activo = True
                print(f"✅ Predictor activo: {len(reservas_por_esp)} especialidades con reservas ML")
        except Exception as e:
            print(f"⚠️ Error obteniendo reservas del predictor: {e}")
            predictor_activo = False
    
    if usar_reservas_predictivas and not predictor_activo:
        print("⚠️ Predictor no disponible, usando reservas por defecto (15%)")
    
    # Configurar degradación por semana (días lejanos se programan menos)
    def get_factor_semana(dias_desde_inicio):
        """Retorna factor de programación según distancia"""
        if dias_desde_inicio <= 4:  # Primera semana
            return 1.0
        elif dias_desde_inicio <= 9:  # Segunda semana
            return 0.85
        else:  # Más de 2 semanas
            return 0.50
    
    estado_slots = {}
    capacidad_total = 0
    capacidad_reservada = 0
    
    for dia in dias_habiles:
        dia_nombre = DIAS_SEMANA[dia.weekday()]
        dias_desde_inicio = (dia - fecha_inicio).days
        factor_semana = get_factor_semana(dias_desde_inicio)
        
        for q in range(1, NUM_QUIROFANOS + 1):
            for turno in TURNOS:
                esp = configuracion_sesiones[q][dia_nombre][turno]
                if esp not in ['LIBRE', 'CERRADO']:
                    h_ini = HORARIO_MANANA_INICIO if turno == 'Mañana' else HORARIO_TARDE_INICIO
                    h_fin = HORARIO_MANANA_FIN if turno == 'Mañana' else HORARIO_TARDE_FIN
                    
                    capacidad_bruta = h_fin - h_ini
                    capacidad_total += capacidad_bruta
                    
                    # Calcular reserva para urgencias
                    reserva_pct = 0
                    if usar_reservas_predictivas and predictor_activo and esp in reservas_por_esp:
                        reserva_pct = reservas_por_esp[esp]['pct_reserva']
                        if dia.weekday() in [0, 1]:
                            reserva_pct *= 1.15
                        elif dia.weekday() == 4:
                            reserva_pct *= 0.85
                    elif usar_reservas_predictivas:
                        reserva_pct = 15.0
                        if dia.weekday() in [0, 1]:
                            reserva_pct *= 1.15
                        elif dia.weekday() == 4:
                            reserva_pct *= 0.85
                    else:
                        reserva_pct = 0
                    
                    reserva_pct = max(reserva_pct, 100 - pct_programacion)
                    minutos_reserva = int(capacidad_bruta * reserva_pct / 100)
                    capacidad_reservada += minutos_reserva
                    capacidad_efectiva = capacidad_bruta - minutos_reserva
                    capacidad_efectiva = int(capacidad_efectiva * factor_semana)
                    hora_fin_efectiva = h_ini + capacidad_efectiva
                    
                    estado_slots[(dia, q, turno)] = {
                        'hora_ini': h_ini, 
                        'hora_fin': hora_fin_efectiva,
                        'hora_fin_real': h_fin,
                        'hora_disp': h_ini, 
                        'esp': esp,
                        'reserva_pct': reserva_pct,
                        'factor_semana': factor_semana
                    }
    
    solicitudes_validas = [s for s in solicitudes if s.activa and not s.cancelada and s.preoperatorio_completado]
    
    # Seleccionar método de asignación
    if metodo == 'genetico' and DEAP_AVAILABLE:
        ids_prog, asignaciones = _optimizar_genetico(solicitudes_validas, estado_slots, cirujanos, fecha_inicio)
    elif metodo == 'milp' and ORTOOLS_AVAILABLE:
        ids_prog, asignaciones = _optimizar_milp(solicitudes_validas, estado_slots, cirujanos, fecha_inicio)
    else:
        # Heurístico (por defecto)
        ids_prog, asignaciones = _optimizar_heuristico(solicitudes_validas, estado_slots, cirujanos, fecha_inicio)
    
    # Aplicar asignaciones al periodo
    motivos_no_prog = {}
    for sol_id, asig in asignaciones.items():
        sol = next((s for s in solicitudes_validas if s.id == sol_id), None)
        if sol:
            cir = CirugiaProgramada(
                id=generar_id_cirugia(), solicitud=sol, fecha=asig['dia'],
                hora_inicio=asig['hora'], duracion_programada_min=asig['duracion'],
                quirofano_id=asig['quirofano'], cirujano=asig['cirujano']
            )
            periodo.obtener_dia(asig['dia']).cirugias.append(cir)
    
    # Calcular motivos de no programación
    for sol in solicitudes_validas:
        if sol.id not in ids_prog:
            esp = sol.tipo_intervencion.especialidad.name
            if esp not in motivos_no_prog:
                motivos_no_prog[esp] = {'sin_sesion': 0, 'sin_hueco': 0}
            tiene_sesion = any(slot['esp'] == esp for slot in estado_slots.values())
            if tiene_sesion:
                motivos_no_prog[esp]['sin_hueco'] += 1
            else:
                motivos_no_prog[esp]['sin_sesion'] += 1
    
    # Métricas
    onco_tot = sum(1 for s in solicitudes_validas if 'ONCOLOGICO' in s.prioridad.name)
    onco_prog = sum(1 for s in solicitudes_validas if s.id in ids_prog and 'ONCOLOGICO' in s.prioridad.name)
    fp_tot = sum(1 for s in solicitudes_validas if s.esta_fuera_plazo)
    fp_prog = sum(1 for s in solicitudes_validas if s.id in ids_prog and s.esta_fuera_plazo)
    
    sc = 0.5 * (onco_prog / max(1, onco_tot)) + 0.3 * (fp_prog / max(1, fp_tot)) + 0.2 * (len(ids_prog) / max(1, len(solicitudes_validas)))
    
    utils = []
    for slot in estado_slots.values():
        total = slot['hora_fin'] - slot['hora_ini']
        usado = slot['hora_disp'] - slot['hora_ini']
        if total > 0:
            utils.append(min(1.0, usado / total))
    se = np.mean(utils) if utils else 0.0
    
    pc = programador.pesos.peso_prioridad_clinica
    pe = programador.pesos.peso_eficiencia_operativa
    
    r = ResultadoOpt()
    r.programa = periodo
    r.score_total = pc * sc + pe * se
    r.score_clinico = sc
    r.score_eficiencia = se
    r.cirugias_programadas = len(ids_prog)
    r.cirugias_no_programadas = len(solicitudes_validas) - len(ids_prog)
    r.robustez_score = 0.7 + 0.3 * (1 - se)
    r.tiempo_ejecucion_seg = time.time() - inicio
    r.reservas_aplicadas = reservas_por_esp
    r.capacidad_efectiva_total = capacidad_total - capacidad_reservada
    r.capacidad_reservada_total = capacidad_reservada
    r.predictor_ml_activo = predictor_activo
    r.motivos_no_programacion = motivos_no_prog
    
    return r


def optimizar_con_config_especifica(solicitudes, cirujanos, config_sesiones, fecha_inicio, horizonte_dias=10):
    """Optimiza usando una configuración de sesiones específica (no la global)"""
    inicio = time.time()
    fecha_fin = fecha_inicio + timedelta(days=horizonte_dias)
    periodo = ProgramaPeriodo(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    
    dias_habiles = [fecha_inicio + timedelta(days=i) for i in range(horizonte_dias + 5)
                   if (fecha_inicio + timedelta(days=i)).weekday() < 5][:horizonte_dias]
    
    estado_slots = {}
    for dia in dias_habiles:
        dia_nombre = DIAS_SEMANA[dia.weekday()]
        for q in range(1, NUM_QUIROFANOS + 1):
            for turno in TURNOS:
                esp = config_sesiones[q][dia_nombre][turno]
                if esp not in ['LIBRE', 'CERRADO']:
                    h_ini = HORARIO_MANANA_INICIO if turno == 'Mañana' else HORARIO_TARDE_INICIO
                    h_fin = HORARIO_MANANA_FIN if turno == 'Mañana' else HORARIO_TARDE_FIN
                    estado_slots[(dia, q, turno)] = {
                        'hora_ini': h_ini, 'hora_fin': h_fin, 'hora_disp': h_ini, 'esp': esp
                    }
    
    solicitudes_validas = [s for s in solicitudes if s.activa and not s.cancelada and s.preoperatorio_completado]
    solicitudes_ord = sorted(solicitudes_validas, key=lambda x: x.score_clinico, reverse=True)
    
    ids_prog = set()
    for sol in solicitudes_ord:
        esp = sol.tipo_intervencion.especialidad.name
        dur = sol.duracion_esperada()
        
        mejor = None
        for (dia, q, turno), slot in estado_slots.items():
            if slot['esp'] != esp:
                continue
            if slot['hora_disp'] + dur + 30 > slot['hora_fin'] + 30:
                continue
            score = 100 - (dia - fecha_inicio).days * 5 + (10 if turno == 'Mañana' else 0)
            if mejor is None or score > mejor[0]:
                mejor = (score, dia, q, turno, slot['hora_disp'])
        
        if mejor:
            _, dia, q, turno, hora = mejor
            if not sol.cirujano_asignado:
                cirs = [c for c in cirujanos if c.especialidad_principal.name == esp]
                if cirs:
                    sol.cirujano_asignado = random.choice(cirs)
            
            cir = CirugiaProgramada(
                id=generar_id_cirugia(), solicitud=sol, fecha=dia,
                hora_inicio=hora, duracion_programada_min=dur,
                quirofano_id=q, cirujano=sol.cirujano_asignado
            )
            periodo.obtener_dia(dia).cirugias.append(cir)
            estado_slots[(dia, q, turno)]['hora_disp'] = hora + dur + 30
            ids_prog.add(sol.id)
    
    # Métricas
    onco_tot = sum(1 for s in solicitudes_validas if 'ONCOLOGICO' in s.prioridad.name)
    onco_prog = sum(1 for s in solicitudes_validas if s.id in ids_prog and 'ONCOLOGICO' in s.prioridad.name)
    fp_tot = sum(1 for s in solicitudes_validas if s.esta_fuera_plazo)
    fp_prog = sum(1 for s in solicitudes_validas if s.id in ids_prog and s.esta_fuera_plazo)
    
    sc = 0.5 * (onco_prog / max(1, onco_tot)) + 0.3 * (fp_prog / max(1, fp_tot)) + 0.2 * (len(ids_prog) / max(1, len(solicitudes_validas)))
    
    utils = []
    for slot in estado_slots.values():
        total = slot['hora_fin'] - slot['hora_ini']
        usado = slot['hora_disp'] - slot['hora_ini']
        if total > 0:
            utils.append(min(1.0, usado / total))
    se = np.mean(utils) if utils else 0.0
    
    pc = programador.pesos.peso_prioridad_clinica
    pe = programador.pesos.peso_eficiencia_operativa
    
    r = ResultadoOpt()
    r.programa = periodo
    r.score_total = pc * sc + pe * se
    r.score_clinico = sc
    r.score_eficiencia = se
    r.cirugias_programadas = len(ids_prog)
    r.cirugias_no_programadas = len(solicitudes_validas) - len(ids_prog)
    r.robustez_score = 0.7 + 0.3 * (1 - se)
    r.tiempo_ejecucion_seg = time.time() - inicio
    return r


# =============================================================================
# CÁLCULO DE CONFIGURACIÓN ÓPTIMA DE SESIONES
# =============================================================================

# Variable global para guardar la configuración óptima calculada
configuracion_optima_calculada = None

def calcular_demanda_por_especialidad():
    """Analiza la lista de espera y calcula la demanda ponderada por especialidad"""
    demanda = {}
    
    solicitudes_validas = [s for s in programador.lista_espera 
                          if s.activa and not s.cancelada and s.preoperatorio_completado]
    
    for sol in solicitudes_validas:
        esp = sol.tipo_intervencion.especialidad.name
        if esp not in demanda:
            demanda[esp] = {
                'cantidad': 0,
                'horas_necesarias': 0,
                'oncologicos': 0,
                'fuera_plazo': 0,
                'score_total': 0,
                'duracion_media': 0
            }
        
        demanda[esp]['cantidad'] += 1
        demanda[esp]['horas_necesarias'] += sol.duracion_esperada() / 60  # en horas
        demanda[esp]['score_total'] += sol.score_clinico
        
        if 'ONCOLOGICO' in sol.prioridad.name:
            demanda[esp]['oncologicos'] += 1
        if sol.esta_fuera_plazo:
            demanda[esp]['fuera_plazo'] += 1
    
    # Calcular duración media y prioridad ponderada
    for esp in demanda:
        if demanda[esp]['cantidad'] > 0:
            demanda[esp]['duracion_media'] = demanda[esp]['horas_necesarias'] / demanda[esp]['cantidad'] * 60
            demanda[esp]['prioridad_ponderada'] = (
                demanda[esp]['score_total'] / demanda[esp]['cantidad'] +
                demanda[esp]['oncologicos'] * 10 +
                demanda[esp]['fuera_plazo'] * 5
            )
        else:
            demanda[esp]['prioridad_ponderada'] = 0
    
    return demanda


def calcular_configuracion_optima():
    """Calcula la distribución óptima de sesiones basada en la demanda y restricciones"""
    global configuracion_optima_calculada
    
    demanda = calcular_demanda_por_especialidad()
    
    if not demanda:
        return None
    
    # Obtener límites de sesiones de las restricciones
    limites_sesiones = obtener_limites_sesiones()
    
    # Calcular horas totales disponibles por semana
    HORAS_MANANA_POR_SESION = 7
    HORAS_TARDE_POR_SESION = 5
    SESIONES_MANANA_DISPONIBLES = NUM_QUIROFANOS * len(DIAS_SEMANA)  # 40
    SESIONES_TARDE_DISPONIBLES = NUM_QUIROFANOS * len(DIAS_SEMANA)   # 40
    
    # Ordenar especialidades por prioridad ponderada (más urgentes primero)
    especialidades_ordenadas = sorted(
        demanda.keys(),
        key=lambda x: demanda[x]['prioridad_ponderada'],
        reverse=True
    )
    
    # Calcular sesiones necesarias por especialidad
    sesiones_necesarias = {}
    for esp in especialidades_ordenadas:
        horas = demanda[esp]['horas_necesarias']
        sesiones_m = int(np.ceil(horas / HORAS_MANANA_POR_SESION))
        
        # Aplicar límites de restricciones
        if esp in limites_sesiones:
            sesiones_m = min(sesiones_m, limites_sesiones[esp]['max'])
            sesiones_m = max(sesiones_m, limites_sesiones[esp]['min'])
        
        sesiones_necesarias[esp] = {
            'horas': horas,
            'sesiones_ideales': sesiones_m,
            'oncologicos': demanda[esp]['oncologicos'],
            'fuera_plazo': demanda[esp]['fuera_plazo']
        }
    
    # Crear nueva configuración base
    nueva_config = {}
    for q in range(1, NUM_QUIROFANOS + 1):
        nueva_config[q] = {}
        for dia in DIAS_SEMANA:
            nueva_config[q][dia] = {'Mañana': 'LIBRE', 'Tarde': 'LIBRE'}
    
    # Aplicar restricciones que cierran slots ANTES de asignar
    nueva_config = aplicar_restricciones_a_config(nueva_config)
    
    # Asignar sesiones de mañana primero (más valiosas)
    sesiones_asignadas = {esp: 0 for esp in especialidades_ordenadas}
    
    # Primera pasada: asignar al menos 1 sesión de mañana a cada especialidad con demanda
    for esp in especialidades_ordenadas:
        if demanda[esp]['cantidad'] > 0:
            # Buscar slot disponible
            for dia in DIAS_SEMANA:
                for q in range(1, NUM_QUIROFANOS + 1):
                    if nueva_config[q][dia]['Mañana'] == 'LIBRE':
                        nueva_config[q][dia]['Mañana'] = esp
                        sesiones_asignadas[esp] += 1
                        break
                if sesiones_asignadas[esp] > 0:
                    break
    
    # Segunda pasada: distribuir sesiones restantes según demanda
    for esp in especialidades_ordenadas:
        max_sesiones = sesiones_necesarias[esp]['sesiones_ideales']
        if esp in limites_sesiones:
            max_sesiones = min(max_sesiones, limites_sesiones[esp]['max'])
        
        while sesiones_asignadas[esp] < max_sesiones:
            asignado = False
            for dia in DIAS_SEMANA:
                for q in range(1, NUM_QUIROFANOS + 1):
                    if nueva_config[q][dia]['Mañana'] == 'LIBRE':
                        nueva_config[q][dia]['Mañana'] = esp
                        sesiones_asignadas[esp] += 1
                        asignado = True
                        break
                if asignado:
                    break
            if not asignado:
                break  # No hay más slots de mañana
    
    # Tercera pasada: usar tardes si es necesario
    for esp in especialidades_ordenadas:
        horas_asignadas = sesiones_asignadas[esp] * HORAS_MANANA_POR_SESION
        horas_faltantes = demanda[esp]['horas_necesarias'] - horas_asignadas
        
        # Verificar si la especialidad puede operar en tardes (restricción solo mañanas)
        puede_tardes = True
        for r in restricciones_usuario:
            if r['tipo'] == 'esp_solo_manana' and r['params'].get('especialidad') == esp:
                puede_tardes = False
                break
        
        if horas_faltantes > 0 and puede_tardes:
            sesiones_tarde = int(np.ceil(horas_faltantes / HORAS_TARDE_POR_SESION))
            
            # Aplicar límite máximo si existe
            if esp in limites_sesiones:
                sesiones_tarde = min(sesiones_tarde, 
                                    limites_sesiones[esp]['max'] - sesiones_asignadas[esp])
            
            tardes_asignadas = 0
            for dia in DIAS_SEMANA:
                for q in range(1, NUM_QUIROFANOS + 1):
                    if nueva_config[q][dia]['Tarde'] == 'LIBRE' and tardes_asignadas < sesiones_tarde:
                        nueva_config[q][dia]['Tarde'] = esp
                        tardes_asignadas += 1
                        sesiones_asignadas[esp] += 1
    
    # Aplicar restricciones finales (por si alguna se perdió)
    nueva_config = aplicar_restricciones_a_config(nueva_config)
    
    configuracion_optima_calculada = nueva_config
    return nueva_config


def comparar_configuraciones(peso_clinico):
    """Compara la configuración actual con la óptima calculada.
    Usa el resultado de la última optimización como baseline si está disponible."""
    global configuracion_optima_calculada, ultimo_resultado_sesiones
    
    programador.configurar_pesos(peso_clinico/100, (100-peso_clinico)/100)
    fecha_inicio = date.today() + timedelta(days=1)
    
    # 1. USAR RESULTADO EXISTENTE como "Config. Actual" si está disponible
    resultado_actual = ultimo_resultado_sesiones
    
    if resultado_actual is None:
        # Si no hay resultado previo, optimizar con config actual
        config_actual_con_restricciones = aplicar_restricciones_a_config(configuracion_sesiones)
        resultado_actual = optimizar_con_config_especifica(
            programador.lista_espera, programador.cirujanos,
            config_actual_con_restricciones, fecha_inicio, 10
        )
        nota_baseline = "*(Optimización nueva - no había resultado previo)*"
    else:
        nota_baseline = "*(Usando resultado de la última optimización)*"
    
    # 2. Calcular y optimizar con configuración ÓPTIMA (ya incluye restricciones)
    config_optima = calcular_configuracion_optima()
    if not config_optima:
        return "❌ No se pudo calcular la configuración óptima", None, None, None
    
    resultado_optimo = optimizar_con_config_especifica(
        programador.lista_espera, programador.cirujanos,
        config_optima, fecha_inicio, 10
    )
    
    # Calcular mejoras
    mejora_programadas = resultado_optimo.cirugias_programadas - resultado_actual.cirugias_programadas
    mejora_clinico = (resultado_optimo.score_clinico - resultado_actual.score_clinico) * 100
    mejora_eficiencia = (resultado_optimo.score_eficiencia - resultado_actual.score_eficiencia) * 100
    
    # Calcular total de pacientes
    total_lista = len(programador.lista_espera) if programador.lista_espera else 0
    cobertura_actual = (resultado_actual.cirugias_programadas / total_lista * 100) if total_lista > 0 else 0
    cobertura_optima = (resultado_optimo.cirugias_programadas / total_lista * 100) if total_lista > 0 else 0
    
    # Generar resumen
    resumen = f"""
## 📊 Comparación de Configuraciones

{nota_baseline}
"""
    
    # Mostrar restricciones activas
    if restricciones_usuario:
        resumen += f"\n### 🚫 Restricciones Aplicadas ({len(restricciones_usuario)})\n"
        for r in restricciones_usuario:
            resumen += f"- {r['texto']}\n"
        resumen += "\n"
    
    resumen += f"""
### Resultados

| Métrica | Config. Actual | Config. Óptima | Diferencia |
|---------|----------------|----------------|------------|
| **Cirugías Programadas** | {resultado_actual.cirugias_programadas} / {total_lista} | {resultado_optimo.cirugias_programadas} / {total_lista} | **{'+' if mejora_programadas >= 0 else ''}{mejora_programadas}** |
| **Cobertura** | {cobertura_actual:.1f}% | {cobertura_optima:.1f}% | {'+' if mejora_programadas >= 0 else ''}{cobertura_optima - cobertura_actual:.1f}% |
| **Score Clínico** | {resultado_actual.score_clinico:.3f} | {resultado_optimo.score_clinico:.3f} | {'+' if mejora_clinico >= 0 else ''}{mejora_clinico:.1f}% |
| **Score Eficiencia** | {resultado_actual.score_eficiencia:.3f} | {resultado_optimo.score_eficiencia:.3f} | {'+' if mejora_eficiencia >= 0 else ''}{mejora_eficiencia:.1f}% |
| **Score Total** | {resultado_actual.score_total:.3f} | {resultado_optimo.score_total:.3f} | {'+' if resultado_optimo.score_total >= resultado_actual.score_total else ''}{(resultado_optimo.score_total - resultado_actual.score_total):.3f} |
| **Pendientes** | {resultado_actual.cirugias_no_programadas} | {resultado_optimo.cirugias_no_programadas} | {resultado_optimo.cirugias_no_programadas - resultado_actual.cirugias_no_programadas} |

### Interpretación
"""
    
    if mejora_programadas > 5:
        resumen += f"\n✅ **Mejora significativa**: La configuración óptima programa **{mejora_programadas} cirugías más**."
    elif mejora_programadas > 0:
        resumen += f"\n⚠️ **Mejora moderada**: La configuración óptima programa {mejora_programadas} cirugías más."
    elif mejora_programadas == 0:
        resumen += f"\n✅ **Configuración actual óptima**: No hay mejora posible con redistribución de sesiones."
    else:
        resumen += f"\n✅ **Configuración actual mejor**: Tu distribución actual es mejor que la calculada automáticamente."
    
    # Análisis de la demanda
    demanda = calcular_demanda_por_especialidad()
    resumen += "\n\n### Análisis de Demanda por Especialidad\n\n"
    resumen += "| Especialidad | Pacientes | Horas Nec. | Oncológicos | Fuera Plazo |\n"
    resumen += "|--------------|-----------|------------|-------------|-------------|\n"
    
    for esp in sorted(demanda.keys(), key=lambda x: demanda[x]['cantidad'], reverse=True)[:10]:
        d = demanda[esp]
        nombre = ESPECIALIDADES_NOMBRES.get(esp, esp)
        resumen += f"| {nombre} | {d['cantidad']} | {d['horas_necesarias']:.1f}h | {d['oncologicos']} | {d['fuera_plazo']} |\n"
    
    # Gráfico comparativo
    fig_comp = go.Figure()
    metricas = ['Cirugías Prog.', 'Score Clínico (%)', 'Score Eficiencia (%)']
    valores_actual = [resultado_actual.cirugias_programadas, resultado_actual.score_clinico*100, resultado_actual.score_eficiencia*100]
    valores_optimo = [resultado_optimo.cirugias_programadas, resultado_optimo.score_clinico*100, resultado_optimo.score_eficiencia*100]
    
    fig_comp.add_trace(go.Bar(name='Configuración Actual', x=metricas, y=valores_actual, marker_color='#3498db'))
    fig_comp.add_trace(go.Bar(name='Configuración Óptima', x=metricas, y=valores_optimo, marker_color='#27ae60'))
    fig_comp.update_layout(barmode='group', title='Comparación de Resultados', height=350)
    
    # Tabla de configuración óptima sugerida
    datos_config = []
    for q in range(1, NUM_QUIROFANOS + 1):
        fila = {'Q': f'Q{q}'}
        for dia in DIAS_SEMANA:
            esp_m = config_optima[q][dia]['Mañana']
            esp_t = config_optima[q][dia]['Tarde']
            fila[f'{dia[:3]}M'] = ESPECIALIDADES_NOMBRES.get(esp_m, esp_m)[:8]
            fila[f'{dia[:3]}T'] = ESPECIALIDADES_NOMBRES.get(esp_t, esp_t)[:8]
        datos_config.append(fila)
    
    df_config = pd.DataFrame(datos_config)
    
    # Gráfico de demanda vs capacidad
    fig_demanda = go.Figure()
    especialidades = list(demanda.keys())[:8]
    horas_demanda = [demanda[e]['horas_necesarias'] for e in especialidades]
    
    # Calcular horas asignadas en config actual y óptima
    def calcular_horas_config(config):
        horas = {esp: 0 for esp in especialidades}
        for q in range(1, NUM_QUIROFANOS + 1):
            for dia in DIAS_SEMANA:
                esp_m = config[q][dia]['Mañana']
                esp_t = config[q][dia]['Tarde']
                if esp_m in horas:
                    horas[esp_m] += 7
                if esp_t in horas:
                    horas[esp_t] += 5
        return [horas[e] for e in especialidades]
    
    horas_actual = calcular_horas_config(configuracion_sesiones)
    horas_optima = calcular_horas_config(config_optima)
    
    nombres_esp = [ESPECIALIDADES_NOMBRES.get(e, e)[:10] for e in especialidades]
    
    fig_demanda.add_trace(go.Bar(name='Demanda (horas)', x=nombres_esp, y=horas_demanda, marker_color='#e74c3c'))
    fig_demanda.add_trace(go.Bar(name='Config. Actual', x=nombres_esp, y=horas_actual, marker_color='#3498db'))
    fig_demanda.add_trace(go.Bar(name='Config. Óptima', x=nombres_esp, y=horas_optima, marker_color='#27ae60'))
    fig_demanda.update_layout(barmode='group', title='Demanda vs Capacidad Asignada (horas/semana)', height=350)
    
    return resumen, fig_comp, df_config, fig_demanda


def aplicar_configuracion_optima():
    """Aplica la configuración óptima calculada a la configuración global"""
    global configuracion_sesiones, configuracion_optima_calculada
    
    if configuracion_optima_calculada is None:
        return "❌ Primero debes ejecutar la comparación para calcular la configuración óptima"
    
    # Copiar configuración óptima a la global
    for q in range(1, NUM_QUIROFANOS + 1):
        for dia in DIAS_SEMANA:
            for turno in TURNOS:
                configuracion_sesiones[q][dia][turno] = configuracion_optima_calculada[q][dia][turno]
    
    # Generar valores para actualizar dropdowns
    valores = []
    for q in range(1, NUM_QUIROFANOS + 1):
        for dia in DIAS_SEMANA:
            for turno in TURNOS:
                valores.append(configuracion_sesiones[q][dia][turno])
    
    return ["✅ **Configuración óptima aplicada.** Ve a la pestaña 'Sesiones' para ver los cambios.", 
            generar_resumen_configuracion(), generar_matriz_visual(), generar_grafico_sesiones()] + valores

def analizar_motivo(sol, ids_prog, cirujanos):
    if not sol.preoperatorio_completado:
        return "⏳ Preoperatorio pendiente"
    if not sol.consentimiento_firmado:
        return "📝 Consentimiento pendiente"
    if sol.cancelada:
        return "❌ Cancelada"
    
    esp = sol.tipo_intervencion.especialidad.name
    tiene_sesion = any(configuracion_sesiones[q][d][t] == esp 
                      for q in range(1, NUM_QUIROFANOS + 1) for d in DIAS_SEMANA for t in TURNOS)
    if not tiene_sesion:
        return f"🗓️ Sin sesión para {ESPECIALIDADES_NOMBRES.get(esp, esp)}"
    return "📅 Capacidad agotada"


def obtener_estadisticas():
    stats = programador.estadisticas_lista_espera()
    resumen = f"## 📊 Lista de Espera\n\n| Métrica | Valor |\n|---------|-------|\n"
    resumen += f"| **Total** | {stats['total_solicitudes']} |\n"
    resumen += f"| **Fuera de plazo** | {stats['fuera_de_plazo']} ({stats['porcentaje_fuera_plazo']:.1f}%) |\n"
    resumen += f"| **Días espera promedio** | {stats['dias_espera_promedio']:.0f} |\n"
    
    prios = list(stats['por_prioridad'].keys())
    fig_prio = go.Figure(data=[go.Bar(
        x=[stats['por_prioridad'][p]['cantidad'] for p in prios],
        y=[p.replace('_', ' ').title() for p in prios],
        orientation='h', marker_color=['#c0392b', '#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#95a5a6'][:len(prios)]
    )])
    fig_prio.update_layout(title='Por Prioridad', height=300)
    
    esps = list(stats['por_especialidad'].keys())
    fig_esp = go.Figure(data=[go.Pie(
        labels=[e.replace('_', ' ').title() for e in esps],
        values=[stats['por_especialidad'][e]['cantidad'] for e in esps], hole=0.4
    )])
    fig_esp.update_layout(title='Por Especialidad', height=300)
    
    return resumen, fig_prio, fig_esp


def optimizar(peso_clinico, metodo, usar_reservas, pct_programacion):
    global ultimo_resultado_sesiones
    programador.configurar_pesos(peso_clinico/100, (100-peso_clinico)/100)
    
    fecha_inicio = date.today() + timedelta(days=1)
    
    # Usar optimizador con sesiones incluyendo reservas predictivas
    # AHORA pasamos el método correctamente
    r = optimizar_con_sesiones(
        programador.lista_espera, 
        programador.cirujanos, 
        fecha_inicio, 
        horizonte_dias=10,
        usar_reservas_predictivas=usar_reservas,
        pct_programacion=pct_programacion,
        metodo=metodo  # ← Ahora se pasa el método
    )
    
    ultimo_resultado_sesiones = r
    programador.ultimo_resultado = r
    
    # Calcular métricas de lista de espera
    total_lista = len([s for s in programador.lista_espera if s.activa and not s.cancelada])
    cobertura_pct = (r.cirugias_programadas / max(1, total_lista)) * 100
    
    # Obtener método real usado
    metodo_usado = getattr(r, 'metodo_usado', metodo if metodo != 'auto' else 'heuristico')
    
    metricas = f"""
## 🎯 Resultado de Optimización

**Método:** {metodo_usado.upper()} | **Peso Clínico:** {peso_clinico}% | **Peso Eficiencia:** {100-peso_clinico}%
"""
    
    # Si es modo auto, mostrar comparativa
    if metodo == 'auto' and hasattr(r, 'comparativa_metodos') and r.comparativa_metodos:
        metricas += f"""
### 🔄 Modo AUTO - Comparativa de Métodos

| Método | Score Total | Score Clínico | Programadas | Tiempo |
|--------|-------------|---------------|-------------|--------|
"""
        for m, datos in r.comparativa_metodos.items():
            es_mejor = "✅" if m == metodo_usado else ""
            metricas += f"| {m.upper()} {es_mejor} | {datos['score_total']:.4f} | {datos['score_clinico']:.4f} | {datos['programadas']} | {datos['tiempo']:.2f}s |\n"
        
        metricas += f"\n**→ Seleccionado: {metodo_usado.upper()}** (mayor score total)\n"
    
    # Info de reservas si están activas
    if usar_reservas:
        horas_reservadas = r.capacidad_reservada_total / 60
        horas_efectivas = r.capacidad_efectiva_total / 60
        total_cap = r.capacidad_efectiva_total + r.capacidad_reservada_total
        pct_reservado = (r.capacidad_reservada_total / total_cap * 100) if total_cap > 0 else 0
        
        if r.predictor_ml_activo and r.reservas_aplicadas:
            metricas += f"""
### 🔮 Reservas Predictivas ML Aplicadas ✅
- **Capacidad reservada para urgencias:** {horas_reservadas:.1f}h ({pct_reservado:.1f}%)
- **Capacidad efectiva para electiva:** {horas_efectivas:.1f}h
"""
            # Top 3 especialidades con más reserva
            top_reservas = sorted(r.reservas_aplicadas.items(), key=lambda x: x[1]['pct_reserva'], reverse=True)[:3]
            metricas += "\n**Mayor reserva (ML):** " + ", ".join([f"{ESPECIALIDADES_NOMBRES.get(e, e)} ({d['pct_reserva']:.0f}%)" for e, d in top_reservas])
            metricas += "\n"
        else:
            metricas += f"""
### ⚠️ Reservas por Defecto (15%)
- **Capacidad reservada para urgencias:** {horas_reservadas:.1f}h ({pct_reservado:.1f}%)
- **Capacidad efectiva para electiva:** {horas_efectivas:.1f}h

> *Predictor ML no disponible. Usando reserva uniforme del 15%.*
"""
    
    metricas += f"""
| Métrica | Valor | Significado |
|---------|-------|-------------|
| **Score Total** | {r.score_total:.4f} | Puntuación global (mayor=mejor) |
| **Score Clínico** | {r.score_clinico:.4f} | Prioritarios atendidos |
| **Score Eficiencia** | {r.score_eficiencia:.4f} | Uso de quirófanos |
| **Programadas** | {r.cirugias_programadas} / {total_lista} | Cobertura: **{cobertura_pct:.1f}%** |
| **Pendientes** | {r.cirugias_no_programadas} | Para siguientes semanas |
| **Robustez** | {r.robustez_score:.1%} | Fiabilidad del programa |
| **Tiempo** | {r.tiempo_ejecucion_seg:.2f}s | Tiempo de cálculo |

### Interpretación
"""
    if r.score_clinico > 0.7:
        metricas += "✅ **Excelente cobertura clínica** - La mayoría de casos prioritarios programados.\n"
    elif r.score_clinico > 0.5:
        metricas += "⚠️ **Cobertura moderada** - Algunos prioritarios pendientes. Considera añadir sesiones.\n"
    else:
        metricas += "❌ **Cobertura baja** - Revisa la configuración de sesiones.\n"
    
    if r.score_eficiencia > 0.75:
        metricas += "✅ **Alta eficiencia** - Buen uso de quirófanos.\n"
    elif r.score_eficiencia > 0.5:
        metricas += "⚠️ **Eficiencia moderada** - Considera activar sesiones de tarde.\n"
    else:
        metricas += "❌ **Baja eficiencia** - Muchos quirófanos infrautilizados.\n"
    
    # Mostrar motivos de no programación si hay muchos pendientes
    if hasattr(r, 'motivos_no_programacion') and r.motivos_no_programacion:
        metricas += "\n### ⚠️ Motivos de No Programación\n"
        for esp, motivos in sorted(r.motivos_no_programacion.items(), key=lambda x: sum(x[1].values()), reverse=True)[:5]:
            nombre = ESPECIALIDADES_NOMBRES.get(esp, esp)
            if motivos['sin_sesion'] > 0:
                metricas += f"- **{nombre}**: {motivos['sin_sesion']} sin sesión asignada\n"
            if motivos['sin_hueco'] > 0:
                metricas += f"- **{nombre}**: {motivos['sin_hueco']} sin hueco disponible\n"
    
    fig_sc = go.Figure(data=[go.Bar(
        x=['Total', 'Clínico', 'Eficiencia'],
        y=[r.score_total, r.score_clinico, r.score_eficiencia],
        marker_color=['#3498db', '#e74c3c', '#27ae60'],
        text=[f'{r.score_total:.2f}', f'{r.score_clinico:.2f}', f'{r.score_eficiencia:.2f}'],
        textposition='outside'
    )])
    fig_sc.update_layout(title='Scores', yaxis_range=[0, 1.1], height=300)
    
    dias_data = []
    if r.programa:
        for f in sorted(r.programa.programas_diarios.keys()):
            n_cirugias = len(r.programa.programas_diarios[f].cirugias)
            dias_data.append({
                'dia': f.strftime('%a %d'), 
                'n': n_cirugias,
                'semana': 'S1' if (f - fecha_inicio).days < 5 else 'S2'
            })
    
    # Gráfico con colores por semana
    colors = ['#3498db' if d['semana'] == 'S1' else '#9b59b6' for d in dias_data]
    fig_dias = go.Figure(data=[go.Bar(
        x=[d['dia'] for d in dias_data], 
        y=[d['n'] for d in dias_data],
        marker_color=colors,
        text=[d['n'] for d in dias_data],
        textposition='outside'
    )])
    fig_dias.update_layout(title='Cirugías por día (Azul=S1, Morado=S2)', height=250)
    
    return metricas, fig_sc, fig_dias


def ver_lista_espera(filtro):
    global ultimo_resultado_sesiones
    
    # Usar el resultado guardado
    resultado = ultimo_resultado_sesiones or programador.ultimo_resultado
    
    if not resultado or not resultado.programa:
        return "⚠️ Ejecuta primero una optimización en la pestaña 'Optimizar'", pd.DataFrame(), go.Figure()
    
    ids_prog = set()
    for prog_dia in resultado.programa.programas_diarios.values():
        for c in prog_dia.cirugias:
            ids_prog.add(c.solicitud.id)
    
    datos, motivos = [], {}
    for s in programador.lista_espera:
        prog = s.id in ids_prog
        if filtro == "Solo programadas" and not prog:
            continue
        if filtro == "Solo NO programadas" and prog:
            continue
        
        estado = "✅" if prog else "❌"
        motivo = "—" if prog else analizar_motivo(s, ids_prog, programador.cirujanos)
        if not prog:
            motivos[motivo] = motivos.get(motivo, 0) + 1
        
        datos.append({
            '#': len(datos)+1, 'Score': f'{s.score_clinico:.0f}',
            'Paciente': s.paciente.nombre[:20], 'Intervención': s.tipo_intervencion.nombre[:22],
            'Esp': ESPECIALIDADES_NOMBRES.get(s.tipo_intervencion.especialidad.name, '')[:8],
            'Días': s.dias_en_espera, 'FP': '⚠️' if s.esta_fuera_plazo else '',
            'Est': estado, 'Motivo': motivo
        })
    
    df = pd.DataFrame(datos) if datos else pd.DataFrame()
    tot, prog_count = len(programador.lista_espera), len(ids_prog)
    
    res = f"## 📋 Lista de Espera\n\n**Total:** {tot} | **Programadas:** {prog_count} | **Pendientes:** {tot-prog_count}\n\n"
    if motivos:
        res += "### Motivos no programación\n"
        for m, c in sorted(motivos.items(), key=lambda x: -x[1]):
            res += f"- {m}: {c}\n"
    
    fig = go.Figure()
    if motivos:
        fig = go.Figure(data=[go.Bar(x=list(motivos.values()), y=list(motivos.keys()), orientation='h')])
        fig.update_layout(height=250, margin=dict(l=180), title='Motivos de No Programación')
    
    return res, df, fig


def ver_programa_dia(dia_idx):
    global ultimo_resultado_sesiones
    
    resultado = ultimo_resultado_sesiones or programador.ultimo_resultado
    
    if not resultado or not resultado.programa:
        return "⚠️ Ejecuta primero una optimización en la pestaña 'Optimizar'", pd.DataFrame()
    
    fechas = sorted(resultado.programa.programas_diarios.keys())
    if not fechas:
        return "Sin días programados", pd.DataFrame()
    
    dia_idx = min(int(dia_idx), len(fechas)-1)
    fecha = fechas[dia_idx]
    prog_dia = resultado.programa.programas_diarios[fecha]
    
    dias_sem = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    dia_nombre = DIAS_SEMANA[fecha.weekday()] if fecha.weekday() < 5 else 'Fin de semana'
    
    res = f"## 📅 {fecha} ({dias_sem[fecha.weekday()]})\n\n**Cirugías:** {len(prog_dia.cirugias)}\n\n"
    
    # Mostrar sesiones configuradas para ese día
    if fecha.weekday() < 5:
        res += "### Sesiones del día\n"
        for q in range(1, NUM_QUIROFANOS + 1):
            esp_m = ESPECIALIDADES_NOMBRES.get(configuracion_sesiones[q][dia_nombre]['Mañana'], '')
            esp_t = ESPECIALIDADES_NOMBRES.get(configuracion_sesiones[q][dia_nombre]['Tarde'], '')
            res += f"- Q{q}: M={esp_m} | T={esp_t}\n"
    
    datos = []
    for c in sorted(prog_dia.cirugias, key=lambda x: (x.quirofano_id, x.hora_inicio)):
        t = 'M' if c.hora_inicio < HORARIO_TARDE_INICIO else 'T'
        datos.append({
            'Hora': c.hora_inicio_str, 'Fin': c.hora_fin_str, 'Q': f'Q{c.quirofano_id}', 'T': t,
            'Paciente': c.solicitud.paciente.nombre[:18],
            'Intervención': c.solicitud.tipo_intervencion.nombre[:22],
            'Prioridad': c.solicitud.prioridad.name.replace('_', ' ')[:12],
            'Cirujano': c.cirujano.nombre[:15] if c.cirujano else 'N/A'
        })
    
    return res, pd.DataFrame(datos) if datos else pd.DataFrame()


# =============================================================================
# VISTA GANTT - PROGRAMA POR QUIRÓFANO (v4.6)
# =============================================================================

# Colores para la vista Gantt
COLORES_GANTT = {
    'CIRUGIA_GENERAL': '#3498db',
    'CIRUGIA_DIGESTIVA': '#2ecc71',
    'CIRUGIA_COLORRECTAL': '#e74c3c',
    'CIRUGIA_HEPATOBILIAR': '#9b59b6',
    'CIRUGIA_MAMA': '#e91e63',
    'CIRUGIA_ENDOCRINA': '#f39c12',
    'CIRUGIA_BARIATRICA': '#00bcd4',
    'UROLOGIA': '#ff9800',
    'GINECOLOGIA': '#ff5722',
    'CIRUGIA_VASCULAR': '#795548',
    'CIRUGIA_PLASTICA': '#607d8b',
    'libre': '#c8e6c9',        # Verde claro - tiempo libre
    'reserva': '#ffe0b2',       # Naranja claro - reserva urgencias
    'urgencia': '#ef5350',      # Rojo - urgencia ocupando
    'limpieza': '#eceff1',      # Gris claro - limpieza entre cirugías
    'cerrado': '#f5f5f5',       # Gris muy claro - sesión cerrada
}

def generar_vista_gantt(dia_idx, mostrar_reservas=True, mostrar_urgencias=True):
    """
    Genera una vista Gantt del programa quirúrgico para un día específico.
    
    Muestra:
    - Cirugías programadas (color por especialidad)
    - Tiempo de limpieza como cola de cada cirugía (color claro rayado)
    - Tiempo libre (verde claro)
    - Reserva para urgencias (naranja rayado)
    - Urgencias diferidas ocupando reserva (rojo)
    """
    global ultimo_resultado_sesiones, urgencias_diferidas
    
    resultado = ultimo_resultado_sesiones or (programador.ultimo_resultado if programador else None)
    
    if not resultado or not resultado.programa:
        return "⚠️ Ejecuta primero una optimización en la pestaña 'Optimizar'", None, pd.DataFrame()
    
    fechas = sorted(resultado.programa.programas_diarios.keys())
    if not fechas:
        return "Sin días programados", None, pd.DataFrame()
    
    dia_idx = min(int(dia_idx), len(fechas) - 1)
    fecha = fechas[dia_idx]
    prog_dia = resultado.programa.programas_diarios[fecha]
    
    dias_sem = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    dia_nombre = DIAS_SEMANA[fecha.weekday()] if fecha.weekday() < 5 else 'Fin de semana'
    
    # Obtener reservas ML si están disponibles
    reservas_por_esp = {}
    if PREDICTOR_DISPONIBLE and predictor_urgencias is not None:
        try:
            reservas_por_esp = predictor_urgencias.obtener_reservas_por_especialidad()
        except:
            pass
    
    # Obtener urgencias programadas para este día
    urgencias_del_dia = [u for u in urgencias_diferidas 
                         if u.estado == EstadoUrgencia.PROGRAMADA and u.fecha_programada == fecha]
    
    # Crear figura Gantt
    fig = go.Figure()
    
    # Datos para la tabla resumen
    datos_tabla = []
    
    # Procesar cada quirófano
    for q in range(1, NUM_QUIROFANOS + 1):
        y_pos = NUM_QUIROFANOS - q + 1  # Invertir para Q1 arriba
        
        # Obtener configuración de sesiones para este día
        esp_manana = configuracion_sesiones[q][dia_nombre]['Mañana'] if fecha.weekday() < 5 else 'CERRADO'
        esp_tarde = configuracion_sesiones[q][dia_nombre]['Tarde'] if fecha.weekday() < 5 else 'CERRADO'
        
        # Cirugías de este quirófano
        cirugias_q = sorted([c for c in prog_dia.cirugias if c.quirofano_id == q], 
                           key=lambda x: x.hora_inicio)
        
        # Urgencias asignadas a este quirófano
        urgencias_q = [u for u in urgencias_del_dia if u.quirofano_asignado == q]
        
        # ========== TURNO MAÑANA ==========
        if esp_manana not in ['LIBRE', 'CERRADO']:
            h_ini = HORARIO_MANANA_INICIO
            h_fin = HORARIO_MANANA_FIN
            duracion_turno = h_fin - h_ini
            
            # Calcular reserva para urgencias
            reserva_pct = reservas_por_esp.get(esp_manana, {}).get('pct_reserva', 15.0) if mostrar_reservas else 0
            # Ajustar por día de semana
            if fecha.weekday() in [0, 1]:
                reserva_pct *= 1.15
            elif fecha.weekday() == 4:
                reserva_pct *= 0.85
            reserva_min = int(duracion_turno * reserva_pct / 100)
            
            # Tiempo efectivo para electiva
            h_fin_efectivo = h_fin - reserva_min
            
            # Tiempo de limpieza entre cirugías
            TIEMPO_LIMPIEZA = 30
            
            # Procesar cirugías del turno mañana
            cirugias_manana = [c for c in cirugias_q if c.hora_inicio < HORARIO_TARDE_INICIO]
            
            for idx_c, c in enumerate(cirugias_manana):
                # Bloque de cirugía
                esp = c.solicitud.tipo_intervencion.especialidad.name
                color = COLORES_GANTT.get(esp, '#999999')
                
                # Calcular color más claro para limpieza
                # Convertir hex a RGB, aclarar, y volver a hex
                def aclarar_color(hex_color, factor=0.6):
                    hex_color = hex_color.lstrip('#')
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    r = int(r + (255 - r) * factor)
                    g = int(g + (255 - g) * factor)
                    b = int(b + (255 - b) * factor)
                    return f'#{r:02x}{g:02x}{b:02x}'
                
                color_limpieza = aclarar_color(color, 0.5)
                
                fig.add_trace(go.Bar(
                    x=[c.duracion_programada_min],
                    y=[y_pos],
                    base=[c.hora_inicio],
                    orientation='h',
                    marker=dict(color=color, line=dict(width=1, color='white')),
                    name=ESPECIALIDADES_NOMBRES.get(esp, esp),
                    showlegend=False,
                    text=f"{c.solicitud.paciente.nombre.split()[0]}<br>{c.duracion_programada_min}min",
                    textposition='inside',
                    textfont=dict(size=9, color='white'),
                    hovertemplate=(
                        f"<b>{c.solicitud.paciente.nombre}</b><br>"
                        f"Intervención: {c.solicitud.tipo_intervencion.nombre}<br>"
                        f"Hora: {c.hora_inicio_str} - {c.hora_fin_str}<br>"
                        f"Duración: {c.duracion_programada_min} min<br>"
                        f"Prioridad: {c.solicitud.prioridad.name.replace('_', ' ')}<br>"
                        f"Cirujano: {c.cirujano.nombre if c.cirujano else 'N/A'}"
                        "<extra></extra>"
                    )
                ))
                
                # Bloque de limpieza (cola de la cirugía)
                # Solo si no es la última cirugía o si hay espacio antes de la reserva
                fin_cirugia = c.hora_inicio + c.duracion_programada_min
                es_ultima = (idx_c == len(cirugias_manana) - 1)
                
                if es_ultima:
                    # Última cirugía: limpieza hasta donde quepa antes de reserva
                    espacio_disponible = h_fin_efectivo - fin_cirugia
                    tiempo_limp = min(TIEMPO_LIMPIEZA, max(0, espacio_disponible))
                else:
                    # No es última: limpieza completa
                    tiempo_limp = TIEMPO_LIMPIEZA
                
                if tiempo_limp > 0:
                    fig.add_trace(go.Bar(
                        x=[tiempo_limp],
                        y=[y_pos],
                        base=[fin_cirugia],
                        orientation='h',
                        marker=dict(
                            color=color_limpieza, 
                            line=dict(width=1, color='white'),
                            pattern=dict(shape='/', solidity=0.3)
                        ),
                        name='Limpieza',
                        showlegend=False,
                        hovertemplate=(
                            f"<b>🧹 Limpieza/Preparación</b><br>"
                            f"Duración: {tiempo_limp} min<br>"
                            f"Después de: {c.solicitud.paciente.nombre}"
                            "<extra></extra>"
                        )
                    ))
                
                datos_tabla.append({
                    'Q': f'Q{q}',
                    'Turno': 'M',
                    'Hora': c.hora_inicio_str,
                    'Paciente': c.solicitud.paciente.nombre[:20],
                    'Intervención': c.solicitud.tipo_intervencion.nombre[:25],
                    'Min': c.duracion_programada_min,
                    'Tipo': 'Electiva'
                })
            
            # Tiempo libre antes de la reserva (después de última cirugía + su limpieza)
            if cirugias_manana:
                ultima = cirugias_manana[-1]
                ultimo_fin = ultima.hora_inicio + ultima.duracion_programada_min + TIEMPO_LIMPIEZA
            else:
                ultimo_fin = h_ini
            
            if ultimo_fin < h_fin_efectivo:
                fig.add_trace(go.Bar(
                    x=[h_fin_efectivo - ultimo_fin],
                    y=[y_pos],
                    base=[ultimo_fin],
                    orientation='h',
                    marker=dict(color=COLORES_GANTT['libre'], line=dict(width=1, color='#a5d6a7')),
                    name='Libre',
                    showlegend=False,
                    hovertemplate=f"<b>Tiempo Libre</b><br>{h_fin_efectivo - ultimo_fin} min disponibles<extra></extra>"
                ))
            
            # Reserva para urgencias (si aplica)
            if mostrar_reservas and reserva_min > 0:
                # Verificar si hay urgencia ocupando este hueco
                urgencia_en_hueco = next((u for u in urgencias_q if u.especialidad == esp_manana), None)
                
                if urgencia_en_hueco and mostrar_urgencias:
                    # Urgencia ocupando parte de la reserva
                    tiempo_urgencia = urgencia_en_hueco.duracion_estimada_min
                    tiempo_reserva_libre = max(0, reserva_min - tiempo_urgencia)
                    
                    # Bloque de urgencia
                    fig.add_trace(go.Bar(
                        x=[tiempo_urgencia],
                        y=[y_pos],
                        base=[h_fin - reserva_min],
                        orientation='h',
                        marker=dict(
                            color=COLORES_GANTT['urgencia'],
                            line=dict(width=2, color='#c62828'),
                            pattern=dict(shape='/')
                        ),
                        name='Urgencia',
                        showlegend=False,
                        text=f"⚡{urgencia_en_hueco.id}",
                        textposition='inside',
                        textfont=dict(size=9, color='white'),
                        hovertemplate=(
                            f"<b>🚨 URGENCIA {urgencia_en_hueco.id}</b><br>"
                            f"Paciente: {urgencia_en_hueco.paciente_nombre}<br>"
                            f"Procedimiento: {urgencia_en_hueco.procedimiento}<br>"
                            f"Duración: {tiempo_urgencia} min<br>"
                            f"Prioridad: {urgencia_en_hueco.prioridad}<br>"
                            f"Horas restantes: {urgencia_en_hueco.horas_restantes:.1f}h"
                            "<extra></extra>"
                        )
                    ))
                    
                    datos_tabla.append({
                        'Q': f'Q{q}',
                        'Turno': 'M',
                        'Hora': f"{(h_fin - reserva_min)//60:02d}:{(h_fin - reserva_min)%60:02d}",
                        'Paciente': urgencia_en_hueco.paciente_nombre[:20],
                        'Intervención': urgencia_en_hueco.procedimiento[:25],
                        'Min': tiempo_urgencia,
                        'Tipo': '🚨 URGENCIA'
                    })
                    
                    # Reserva restante
                    if tiempo_reserva_libre > 0:
                        fig.add_trace(go.Bar(
                            x=[tiempo_reserva_libre],
                            y=[y_pos],
                            base=[h_fin - tiempo_reserva_libre],
                            orientation='h',
                            marker=dict(
                                color=COLORES_GANTT['reserva'],
                                line=dict(width=1, color='#ffb74d'),
                                pattern=dict(shape='/')
                            ),
                            name='Reserva Urgencias',
                            showlegend=False,
                            hovertemplate=f"<b>Reserva Urgencias</b><br>{tiempo_reserva_libre} min libres<extra></extra>"
                        ))
                else:
                    # Bloque de reserva completo
                    fig.add_trace(go.Bar(
                        x=[reserva_min],
                        y=[y_pos],
                        base=[h_fin - reserva_min],
                        orientation='h',
                        marker=dict(
                            color=COLORES_GANTT['reserva'],
                            line=dict(width=1, color='#ffb74d'),
                            pattern=dict(shape='/')
                        ),
                        name='Reserva Urgencias',
                        showlegend=False,
                        text=f"Reserva<br>{reserva_min}min",
                        textposition='inside',
                        textfont=dict(size=8, color='#e65100'),
                        hovertemplate=(
                            f"<b>🔶 Reserva para Urgencias</b><br>"
                            f"Especialidad: {ESPECIALIDADES_NOMBRES.get(esp_manana, esp_manana)}<br>"
                            f"Tiempo reservado: {reserva_min} min ({reserva_pct:.0f}%)<br>"
                            f"Basado en: Predicción ML"
                            "<extra></extra>"
                        )
                    ))
        
        elif esp_manana == 'LIBRE':
            # Sesión libre - mostrar como disponible
            fig.add_trace(go.Bar(
                x=[HORARIO_MANANA_FIN - HORARIO_MANANA_INICIO],
                y=[y_pos],
                base=[HORARIO_MANANA_INICIO],
                orientation='h',
                marker=dict(color='#e8f5e9', line=dict(width=1, color='#c8e6c9')),
                name='Sesión Libre',
                showlegend=False,
                hovertemplate=f"<b>Sesión LIBRE</b><br>Mañana: Disponible para asignar<extra></extra>"
            ))
        
        # ========== TURNO TARDE ==========
        if esp_tarde not in ['LIBRE', 'CERRADO']:
            h_ini_t = HORARIO_TARDE_INICIO
            h_fin_t = HORARIO_TARDE_FIN
            duracion_turno_t = h_fin_t - h_ini_t
            
            reserva_pct_t = reservas_por_esp.get(esp_tarde, {}).get('pct_reserva', 15.0) if mostrar_reservas else 0
            if fecha.weekday() in [0, 1]:
                reserva_pct_t *= 1.15
            elif fecha.weekday() == 4:
                reserva_pct_t *= 0.85
            reserva_min_t = int(duracion_turno_t * reserva_pct_t / 100)
            h_fin_efectivo_t = h_fin_t - reserva_min_t
            
            cirugias_tarde = [c for c in cirugias_q if c.hora_inicio >= HORARIO_TARDE_INICIO]
            
            for idx_c, c in enumerate(cirugias_tarde):
                esp = c.solicitud.tipo_intervencion.especialidad.name
                color = COLORES_GANTT.get(esp, '#999999')
                
                # Calcular color más claro para limpieza
                def aclarar_color_t(hex_color, factor=0.6):
                    hex_color = hex_color.lstrip('#')
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    r = int(r + (255 - r) * factor)
                    g = int(g + (255 - g) * factor)
                    b = int(b + (255 - b) * factor)
                    return f'#{r:02x}{g:02x}{b:02x}'
                
                color_limpieza_t = aclarar_color_t(color, 0.5)
                
                fig.add_trace(go.Bar(
                    x=[c.duracion_programada_min],
                    y=[y_pos],
                    base=[c.hora_inicio],
                    orientation='h',
                    marker=dict(color=color, line=dict(width=1, color='white')),
                    name=ESPECIALIDADES_NOMBRES.get(esp, esp),
                    showlegend=False,
                    text=f"{c.solicitud.paciente.nombre.split()[0]}<br>{c.duracion_programada_min}min",
                    textposition='inside',
                    textfont=dict(size=9, color='white'),
                    hovertemplate=(
                        f"<b>{c.solicitud.paciente.nombre}</b><br>"
                        f"Intervención: {c.solicitud.tipo_intervencion.nombre}<br>"
                        f"Hora: {c.hora_inicio_str} - {c.hora_fin_str}<br>"
                        f"Duración: {c.duracion_programada_min} min"
                        "<extra></extra>"
                    )
                ))
                
                # Bloque de limpieza (cola de la cirugía)
                fin_cirugia_t = c.hora_inicio + c.duracion_programada_min
                es_ultima_t = (idx_c == len(cirugias_tarde) - 1)
                
                if es_ultima_t:
                    espacio_disponible_t = h_fin_efectivo_t - fin_cirugia_t
                    tiempo_limp_t = min(TIEMPO_LIMPIEZA, max(0, espacio_disponible_t))
                else:
                    tiempo_limp_t = TIEMPO_LIMPIEZA
                
                if tiempo_limp_t > 0:
                    fig.add_trace(go.Bar(
                        x=[tiempo_limp_t],
                        y=[y_pos],
                        base=[fin_cirugia_t],
                        orientation='h',
                        marker=dict(
                            color=color_limpieza_t, 
                            line=dict(width=1, color='white'),
                            pattern=dict(shape='/', solidity=0.3)
                        ),
                        name='Limpieza',
                        showlegend=False,
                        hovertemplate=(
                            f"<b>🧹 Limpieza/Preparación</b><br>"
                            f"Duración: {tiempo_limp_t} min<br>"
                            f"Después de: {c.solicitud.paciente.nombre}"
                            "<extra></extra>"
                        )
                    ))
                
                datos_tabla.append({
                    'Q': f'Q{q}',
                    'Turno': 'T',
                    'Hora': c.hora_inicio_str,
                    'Paciente': c.solicitud.paciente.nombre[:20],
                    'Intervención': c.solicitud.tipo_intervencion.nombre[:25],
                    'Min': c.duracion_programada_min,
                    'Tipo': 'Electiva'
                })
            
            # Tiempo libre tarde (después de última cirugía + su limpieza)
            if cirugias_tarde:
                ultima_t = cirugias_tarde[-1]
                ultimo_fin_t = ultima_t.hora_inicio + ultima_t.duracion_programada_min + TIEMPO_LIMPIEZA
            else:
                ultimo_fin_t = h_ini_t
            
            if ultimo_fin_t < h_fin_efectivo_t:
                fig.add_trace(go.Bar(
                    x=[h_fin_efectivo_t - ultimo_fin_t],
                    y=[y_pos],
                    base=[ultimo_fin_t],
                    orientation='h',
                    marker=dict(color=COLORES_GANTT['libre'], line=dict(width=1, color='#a5d6a7')),
                    name='Libre',
                    showlegend=False,
                    hovertemplate=f"<b>Tiempo Libre</b><br>{h_fin_efectivo_t - ultimo_fin_t} min disponibles<extra></extra>"
                ))
            
            # Reserva tarde
            if mostrar_reservas and reserva_min_t > 0:
                fig.add_trace(go.Bar(
                    x=[reserva_min_t],
                    y=[y_pos],
                    base=[h_fin_t - reserva_min_t],
                    orientation='h',
                    marker=dict(
                        color=COLORES_GANTT['reserva'],
                        line=dict(width=1, color='#ffb74d'),
                        pattern=dict(shape='/')
                    ),
                    name='Reserva',
                    showlegend=False,
                    hovertemplate=f"<b>Reserva Urgencias</b><br>{reserva_min_t} min<extra></extra>"
                ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=f"📅 Programa Quirúrgico - {dias_sem[fecha.weekday()]} {fecha.strftime('%d/%m/%Y')}",
            font=dict(size=16)
        ),
        barmode='overlay',
        height=100 + NUM_QUIROFANOS * 70,
        xaxis=dict(
            title='Hora del día',
            tickmode='array',
            tickvals=[480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140, 1200],
            ticktext=['08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', 
                     '15:00', '16:00', '17:00', '18:00', '19:00', '20:00'],
            range=[450, 1230],
            gridcolor='#e0e0e0',
            showgrid=True
        ),
        yaxis=dict(
            title='Quirófano',
            tickmode='array',
            tickvals=list(range(1, NUM_QUIROFANOS + 1)),
            ticktext=[f'Q{NUM_QUIROFANOS - i + 1}' for i in range(1, NUM_QUIROFANOS + 1)],
            range=[0.5, NUM_QUIROFANOS + 0.5]
        ),
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor='white',
        # Añadir líneas verticales para separar turnos
        shapes=[
            # Línea separadora Mañana/Tarde
            dict(
                type='line',
                x0=HORARIO_TARDE_INICIO, x1=HORARIO_TARDE_INICIO,
                y0=0.5, y1=NUM_QUIROFANOS + 0.5,
                line=dict(color='#1565c0', width=2, dash='dash')
            ),
            # Línea fin de jornada mañana
            dict(
                type='line',
                x0=HORARIO_MANANA_FIN, x1=HORARIO_MANANA_FIN,
                y0=0.5, y1=NUM_QUIROFANOS + 0.5,
                line=dict(color='#e0e0e0', width=1)
            )
        ],
        # Añadir anotaciones
        annotations=[
            dict(x=690, y=NUM_QUIROFANOS + 0.7, text="MAÑANA", showarrow=False, 
                 font=dict(size=12, color='#1565c0')),
            dict(x=1050, y=NUM_QUIROFANOS + 0.7, text="TARDE", showarrow=False,
                 font=dict(size=12, color='#1565c0'))
        ]
    )
    
    # Generar resumen
    total_cirugias = len(prog_dia.cirugias)
    total_urgencias = len([u for u in urgencias_del_dia])
    total_minutos = sum(c.duracion_programada_min for c in prog_dia.cirugias)
    
    resumen = f"""
## 📅 {dias_sem[fecha.weekday()]} {fecha.strftime('%d/%m/%Y')}

### Resumen del Día
| Métrica | Valor |
|---------|-------|
| **Cirugías electivas** | {total_cirugias} |
| **Urgencias programadas** | {total_urgencias} |
| **Tiempo total programado** | {total_minutos // 60}h {total_minutos % 60}min |
| **Quirófanos activos** | {len(set(c.quirofano_id for c in prog_dia.cirugias))} |

### Leyenda
- 🟦 **Colores**: Especialidad de la cirugía
- 🟩 **Verde claro**: Tiempo libre disponible
- 🟧 **Naranja rayado**: Reserva para urgencias (ML)
- 🟥 **Rojo**: Urgencia diferida ocupando reserva
"""
    
    return resumen, fig, pd.DataFrame(datos_tabla)


def ver_ml():
    if not restricciones_ml:
        return "⚠️ ML no disponible", None, None
    
    res = f"## 🤖 {len(restricciones_ml)} Restricciones ML\n"
    datos = [{'Tipo': r.tipo, 'Descripción': r.descripcion[:50], 'Conf': f'{r.confianza:.0%}'} 
             for r in restricciones_ml[:12]]
    
    fig = go.Figure(data=[go.Bar(
        x=[r.confianza*100 for r in restricciones_ml[:8]],
        y=[r.descripcion[:30] for r in restricciones_ml[:8]], orientation='h'
    )])
    fig.update_layout(height=250, margin=dict(l=200))
    
    return res, pd.DataFrame(datos), fig


# =============================================================================
# FUNCIONES DEL PREDICTOR DE URGENCIAS
# =============================================================================

def ver_predictor_resumen():
    """Genera el resumen del predictor de urgencias"""
    if not PREDICTOR_DISPONIBLE or predictor_urgencias is None:
        return "⚠️ **Predictor no disponible.** Asegúrese de que `urgencias_predictor.py` esté en la carpeta.", None, None, None
    
    # Obtener reservas
    reservas = predictor_urgencias.obtener_reservas_por_especialidad()
    
    # Markdown resumen
    md = f"""## 🔮 Predictor de Urgencias Entrenado
    
**Modelos entrenados:** {len(predictor_urgencias.modelos)}  
**Cirugías analizadas:** {predictor_urgencias.estadisticas_globales['total_cirugias']:,}  
**Urgencias en histórico:** {predictor_urgencias.estadisticas_globales['total_urgencias']:,} ({predictor_urgencias.estadisticas_globales['tasa_urgencias_global']:.1%})  
**Duración media urgencia:** {predictor_urgencias.estadisticas_globales['duracion_media_urgencia']:.0f} min

---

### 📊 Interpretación de los Resultados

| Tasa Urgencias | Significado | Reserva Típica |
|----------------|-------------|----------------|
| > 30% | 🔴 Muy alta (Cir. General típica) | 30-45% |
| 15-30% | 🟡 Media-Alta | 15-30% |
| 5-15% | 🟢 Media | 10-20% |
| < 5% | ⚪ Baja (Mama, Bariátrica) | 5-10% |

> **💡 Nota:** Las reservas se calculan usando ML (Random Forest / Gradient Boosting) 
> entrenado con el histórico de cirugías. Se ajustan por día de semana y mes.
"""
    
    # DataFrame con reservas
    datos = []
    for esp in sorted(reservas.keys(), key=lambda x: reservas[x]['tasa_urgencias'], reverse=True):
        r = reservas[esp]
        # Emoji según tasa
        if r['tasa_urgencias'] > 0.30:
            emoji = "🔴"
        elif r['tasa_urgencias'] > 0.15:
            emoji = "🟡"
        elif r['tasa_urgencias'] > 0.05:
            emoji = "🟢"
        else:
            emoji = "⚪"
        
        datos.append({
            '': emoji,
            'Especialidad': ESPECIALIDADES_NOMBRES.get(esp, esp),
            'Tasa Urg.': f"{r['tasa_urgencias']:.1%}",
            'Reserva %': f"{r['pct_reserva']:.1f}%",
            'Min Mañana': f"{r['minutos_reserva_manana']} min",
            'Min Tarde': f"{r['minutos_reserva_tarde']} min",
            'Confianza': f"{r['confianza']:.0f}%",
            'Modelo': r['modelo']
        })
    
    df = pd.DataFrame(datos)
    
    # Gráfico de barras de reservas
    esp_nombres = [ESPECIALIDADES_NOMBRES.get(e, e) for e in reservas.keys()]
    pcts = [reservas[e]['pct_reserva'] for e in reservas.keys()]
    tasas = [reservas[e]['tasa_urgencias'] * 100 for e in reservas.keys()]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Reserva Recomendada (%)", "Tasa de Urgencias (%)"))
    
    # Ordenar por reserva
    indices = sorted(range(len(pcts)), key=lambda i: pcts[i], reverse=True)
    esp_ord = [esp_nombres[i] for i in indices]
    pcts_ord = [pcts[i] for i in indices]
    tasas_ord = [tasas[i] for i in indices]
    
    colors = ['#ff6b6b' if p > 25 else '#ffd93d' if p > 15 else '#6bcb77' for p in pcts_ord]
    
    fig.add_trace(
        go.Bar(x=pcts_ord, y=esp_ord, orientation='h', marker_color=colors, name='Reserva %'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=tasas_ord, y=esp_ord, orientation='h', marker_color='#4ecdc4', name='Tasa Urg %'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, margin=dict(l=150))
    
    # Gráfico de patrones semanales
    dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie']
    fig2 = go.Figure()
    
    # Top 4 especialidades con más urgencias
    top_esp = sorted(reservas.keys(), key=lambda x: reservas[x]['tasa_urgencias'], reverse=True)[:4]
    
    colors_lineas = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, esp in enumerate(top_esp):
        if esp in predictor_urgencias.modelos:
            patron = predictor_urgencias.modelos[esp].patron_semanal
            valores = [patron.get(d, 1.0) for d in range(5)]
            fig2.add_trace(go.Scatter(
                x=dias, y=valores, mode='lines+markers',
                name=ESPECIALIDADES_NOMBRES.get(esp, esp),
                line=dict(color=colors_lineas[i], width=2)
            ))
    
    fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Media")
    fig2.update_layout(
        title="Variación Semanal (factor vs media)",
        height=300,
        yaxis_title="Factor",
        xaxis_title="Día de semana"
    )
    
    return md, df, fig, fig2


def predecir_semana(fecha_inicio_str):
    """Genera predicciones para los próximos días"""
    if not PREDICTOR_DISPONIBLE or predictor_urgencias is None:
        return "⚠️ Predictor no disponible", None, None
    
    # Parsear fecha o usar mañana
    try:
        if fecha_inicio_str:
            fecha_inicio = datetime.strptime(fecha_inicio_str, "%Y-%m-%d").date()
        else:
            fecha_inicio = date.today() + timedelta(days=1)
    except:
        fecha_inicio = date.today() + timedelta(days=1)
    
    # Asegurar día laborable
    while fecha_inicio.weekday() >= 5:
        fecha_inicio += timedelta(days=1)
    
    dias_nombres = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    
    # Obtener predicciones para cada especialidad
    datos = []
    especialidades = list(predictor_urgencias.modelos.keys())[:8]  # Top 8
    
    for esp in especialidades:
        fila = {'Especialidad': ESPECIALIDADES_NOMBRES.get(esp, esp)}
        fecha = fecha_inicio
        for i in range(5):  # 5 días
            while fecha.weekday() >= 5:
                fecha += timedelta(days=1)
            
            pred = predictor_urgencias.predecir(esp, fecha)
            dia_str = f"{dias_nombres[fecha.weekday()]} {fecha.day}"
            fila[dia_str] = f"{pred.pct_reserva_recomendado:.0f}%"
            fecha += timedelta(days=1)
        datos.append(fila)
    
    df = pd.DataFrame(datos)
    
    # Markdown
    md = f"""## 📅 Predicción: Semana del {fecha_inicio.strftime('%d/%m/%Y')}

Las reservas mostradas son el **% de la sesión** que se recomienda dejar libre para urgencias diferidas.

> **Ejemplo:** Si Cir. General muestra 35%, en una sesión de mañana (420 min) reservar ~147 min.
"""
    
    # Heatmap de predicciones
    fig = go.Figure()
    
    z = []
    y_labels = []
    x_labels = list(df.columns)[1:]  # Sin la columna Especialidad
    
    for _, row in df.iterrows():
        z.append([float(row[col].replace('%', '')) for col in x_labels])
        y_labels.append(row['Especialidad'])
    
    fig.add_trace(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn_r',
        text=[[f"{v:.0f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Especialidad: %{y}<br>Día: %{x}<br>Reserva: %{z:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Mapa de Reservas Recomendadas (%)",
        height=350,
        xaxis_title="Día",
        yaxis_title="Especialidad"
    )
    
    return md, df, fig


def aplicar_reservas_predichas():
    """Aplica las reservas predichas a la configuración actual"""
    global configuracion_sesiones
    
    if not PREDICTOR_DISPONIBLE or predictor_urgencias is None:
        return "⚠️ Predictor no disponible"
    
    # Obtener reservas
    reservas = predictor_urgencias.obtener_reservas_por_especialidad()
    
    # Crear mensaje de resumen
    cambios = []
    for esp, datos in reservas.items():
        if esp in LISTA_ESPECIALIDADES:
            cambios.append(f"- **{ESPECIALIDADES_NOMBRES.get(esp, esp)}**: {datos['pct_reserva']:.1f}% reserva")
    
    msg = f"""## ✅ Reservas Predictivas Calculadas

Las siguientes reservas se aplicarán automáticamente cuando se ejecute la optimización:

{chr(10).join(cambios[:10])}

> **Nota:** El optimizador usará estas reservas para calcular la capacidad efectiva de cada sesión.
> Una sesión de Cirugía General con 38% de reserva tendrá solo 62% de capacidad para cirugía electiva.

### Cómo se aplica:

1. Ve a la pestaña **⚙️ Optimizar**
2. Marca la opción **"Aplicar reservas predictivas"**
3. Ejecuta la optimización

El sistema descontará automáticamente el tiempo de reserva de cada sesión según la especialidad y el día.
"""
    
    return msg


def obtener_reserva_para_optimizador(especialidad, dia_semana):
    """Función auxiliar para el optimizador"""
    if not PREDICTOR_DISPONIBLE or predictor_urgencias is None:
        return 15.0  # Default 15%
    
    if especialidad not in predictor_urgencias.modelos:
        return 15.0
    
    modelo = predictor_urgencias.modelos[especialidad]
    
    # Calcular reserva ajustada por día de semana
    tasa_base = modelo.tasa_urgencias_media
    factor_dia = modelo.patron_semanal.get(dia_semana, 1.0)
    
    reserva = tasa_base * factor_dia * 100
    
    # Limitar entre 5% y 50%
    return max(5.0, min(50.0, reserva))


# =============================================================================
# FUNCIONES DE URGENCIAS DIFERIDAS (UI)
# =============================================================================

def ver_urgencias_pendientes():
    """Muestra la lista de urgencias pendientes y programadas"""
    global urgencias_diferidas
    
    pendientes = [u for u in urgencias_diferidas if u.estado == EstadoUrgencia.PENDIENTE]
    programadas = [u for u in urgencias_diferidas if u.estado == EstadoUrgencia.PROGRAMADA]
    
    # Ordenar pendientes por criticidad
    pendientes.sort(key=lambda u: (u.prioridad, u.horas_restantes))
    
    if not pendientes and not programadas:
        return "✅ **No hay urgencias registradas.**", pd.DataFrame(), None
    
    # Markdown resumen
    criticas = sum(1 for u in pendientes if u.es_critica)
    
    md = f"""## ⚡ Urgencias Diferidas

| Estado | Cantidad |
|--------|----------|
| 🔴 **Pendientes** | {len(pendientes)} |
| ✅ **Programadas** | {len(programadas)} |
| 🚨 **Críticas** (< 12h) | {criticas} |

### Leyenda de Prioridades
| Prioridad | Significado | Tiempo Límite |
|-----------|-------------|---------------|
| 1 | 🔴 Máxima | 24h |
| 2 | 🟠 Alta | 48h |
| 3 | 🟡 Media | 72h |
"""
    
    # DataFrame combinado
    datos = []
    
    # Primero las pendientes
    for u in pendientes:
        horas = u.horas_restantes
        if horas < 12:
            estado_tiempo = f"🚨 {horas:.0f}h"
        elif horas < 24:
            estado_tiempo = f"⚠️ {horas:.0f}h"
        else:
            estado_tiempo = f"✅ {horas:.0f}h"
        
        prio_emoji = {1: "🔴", 2: "🟠", 3: "🟡"}.get(u.prioridad, "⚪")
        
        datos.append({
            'ID': u.id,
            'Estado': '⏳ Pendiente',
            'P': prio_emoji,
            'Paciente': u.paciente_nombre[:20],
            'Edad': u.edad,
            'Especialidad': ESPECIALIDADES_NOMBRES.get(u.especialidad, u.especialidad)[:12],
            'Diagnóstico': u.diagnostico[:20],
            'Procedimiento': u.procedimiento[:18],
            'Duración': f"{u.duracion_estimada_min} min",
            'Restante': estado_tiempo,
            'Asignación': '-'
        })
    
    # Luego las programadas
    for u in programadas:
        prio_emoji = {1: "🔴", 2: "🟠", 3: "🟡"}.get(u.prioridad, "⚪")
        fecha_str = u.fecha_programada.strftime('%a %d/%m') if u.fecha_programada else '-'
        asignacion = f"{fecha_str} Q{u.quirofano_asignado}" if u.quirofano_asignado else fecha_str
        
        datos.append({
            'ID': u.id,
            'Estado': '📅 Programada',
            'P': prio_emoji,
            'Paciente': u.paciente_nombre[:20],
            'Edad': u.edad,
            'Especialidad': ESPECIALIDADES_NOMBRES.get(u.especialidad, u.especialidad)[:12],
            'Diagnóstico': u.diagnostico[:20],
            'Procedimiento': u.procedimiento[:18],
            'Duración': f"{u.duracion_estimada_min} min",
            'Restante': '-',
            'Asignación': asignacion
        })
    
    df = pd.DataFrame(datos)
    
    # Gráfico de barras horizontales por tiempo restante (solo pendientes)
    if pendientes:
        fig = go.Figure()
        
        colors = []
        for u in pendientes:
            if u.horas_restantes < 12:
                colors.append('#e74c3c')  # Rojo crítico
            elif u.horas_restantes < 24:
                colors.append('#f39c12')  # Naranja
            else:
                colors.append('#27ae60')  # Verde
        
        fig.add_trace(go.Bar(
            y=[f"{u.paciente_nombre[:15]} ({u.id})" for u in pendientes],
            x=[u.horas_restantes for u in pendientes],
            orientation='h',
            marker_color=colors,
            text=[f"{u.horas_restantes:.0f}h" for u in pendientes],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="⏳ Urgencias PENDIENTES - Horas Restantes",
            xaxis_title="Horas hasta límite",
            height=max(250, len(pendientes) * 40),
            margin=dict(l=150)
        )
        
        # Añadir línea de 12h (crítico)
        fig.add_vline(x=12, line_dash="dash", line_color="red", 
                      annotation_text="Crítico (<12h)")
    else:
        fig = None
    
    return md, df, fig
    
    return md, df, fig


def agregar_urgencia_ui(paciente, edad, especialidad, diagnostico, 
                         procedimiento, duracion, horas_limite, prioridad, notas):
    """Añade una urgencia desde la UI"""
    if not paciente or not diagnostico or not procedimiento:
        return "⚠️ Complete los campos obligatorios (Paciente, Diagnóstico, Procedimiento)", None, None, None
    
    try:
        urgencia = agregar_urgencia(
            paciente=paciente,
            edad=int(edad),
            especialidad=especialidad,
            diagnostico=diagnostico,
            procedimiento=procedimiento,
            duracion=int(duracion),
            horas_limite=int(horas_limite),
            prioridad=int(prioridad),
            notas=notas or ""
        )
        
        msg = f"✅ **Urgencia {urgencia.id} registrada**\n\n- Paciente: {paciente}\n- Límite: {urgencia.fecha_limite}\n- Horas restantes: {urgencia.horas_restantes:.0f}h"
        
        # Actualizar lista
        md, df, fig = ver_urgencias_pendientes()
        
        return msg, md, df, fig
    
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None


def marcar_urgencia_operada(urgencia_id):
    """Marca una urgencia como operada"""
    for u in urgencias_diferidas:
        if u.id == urgencia_id:
            u.estado = EstadoUrgencia.OPERADA
            md, df, fig = ver_urgencias_pendientes()
            return f"✅ Urgencia {urgencia_id} marcada como OPERADA", md, df, fig
    
    return f"⚠️ No se encontró la urgencia {urgencia_id}", None, None, None


def cancelar_urgencia(urgencia_id):
    """Cancela una urgencia"""
    for u in urgencias_diferidas:
        if u.id == urgencia_id:
            u.estado = EstadoUrgencia.CANCELADA
            md, df, fig = ver_urgencias_pendientes()
            return f"✅ Urgencia {urgencia_id} CANCELADA", md, df, fig
    
    return f"⚠️ No se encontró la urgencia {urgencia_id}", None, None, None


def reset_urgencias_ejemplo():
    """Regenera las urgencias de ejemplo"""
    generar_urgencias_ejemplo()
    md, df, fig = ver_urgencias_pendientes()
    return "✅ Urgencias de ejemplo regeneradas", md, df, fig


# =============================================================================
# FUNCIONES DE VISTA POR SEMANAS (CALENDARIO)
# =============================================================================

def generar_vista_calendario():
    """Genera una vista de calendario semanal del programa"""
    global ultimo_resultado_sesiones
    
    resultado = ultimo_resultado_sesiones or programador.ultimo_resultado
    
    if not resultado or not resultado.programa:
        return "⚠️ **Ejecuta primero una optimización en la pestaña 'Optimizar'**", None, None
    
    # Agrupar cirugías por día
    cirugias_por_dia = defaultdict(list)
    for fecha, prog_dia in resultado.programa.programas_diarios.items():
        for c in prog_dia.cirugias:
            cirugias_por_dia[fecha].append(c)
    
    if not cirugias_por_dia:
        return "⚠️ No hay cirugías programadas", None, None
    
    # Calcular métricas por semana
    fecha_inicio = min(cirugias_por_dia.keys())
    
    semana1_cirugias = 0
    semana2_cirugias = 0
    
    for fecha, cirugias in cirugias_por_dia.items():
        dias_desde_inicio = (fecha - fecha_inicio).days
        if dias_desde_inicio < 5:
            semana1_cirugias += len(cirugias)
        else:
            semana2_cirugias += len(cirugias)
    
    md = f"""## 📅 Vista Calendario del Programa

| Semana | Cirugías | Días Programados |
|--------|----------|------------------|
| **Semana 1** | {semana1_cirugias} | {min(5, len([d for d in cirugias_por_dia.keys() if (d - fecha_inicio).days < 5]))} |
| **Semana 2** | {semana2_cirugias} | {len([d for d in cirugias_por_dia.keys() if (d - fecha_inicio).days >= 5])} |
| **Total** | {semana1_cirugias + semana2_cirugias} | {len(cirugias_por_dia)} |
"""
    
    # Crear matriz para heatmap: filas = quirófanos, columnas = días
    fechas_ordenadas = sorted(cirugias_por_dia.keys())
    matriz = np.zeros((NUM_QUIROFANOS, len(fechas_ordenadas)))
    
    for j, fecha in enumerate(fechas_ordenadas):
        for c in cirugias_por_dia[fecha]:
            q = c.quirofano_id - 1
            if 0 <= q < NUM_QUIROFANOS:
                matriz[q, j] += 1
    
    # Heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=matriz,
        x=[f.strftime('%a %d') for f in fechas_ordenadas],
        y=[f'Q{i+1}' for i in range(NUM_QUIROFANOS)],
        colorscale='Blues',
        text=matriz.astype(int),
        texttemplate="%{text}",
        hovertemplate="Q%{y} - %{x}: %{z} cirugías<extra></extra>"
    ))
    
    fig_heatmap.update_layout(
        title="Cirugías por Quirófano y Día",
        xaxis_title="Día",
        yaxis_title="Quirófano",
        height=350
    )
    
    # Timeline por día
    timeline_data = []
    for fecha in fechas_ordenadas:
        for c in sorted(cirugias_por_dia[fecha], key=lambda x: x.hora_inicio):
            esp = c.solicitud.tipo_intervencion.especialidad.name
            color = {
                'CIRUGIA_GENERAL': '#3498db',
                'CIRUGIA_DIGESTIVA': '#2ecc71',
                'UROLOGIA': '#f39c12',
                'GINECOLOGIA': '#e74c3c',
                'CIRUGIA_MAMA': '#e91e63',
                'CIRUGIA_VASCULAR': '#9b59b6',
            }.get(esp, '#95a5a6')
            
            timeline_data.append({
                'Task': f"Q{c.quirofano_id}",
                'Start': f"{fecha} {c.hora_inicio_str}",
                'Finish': f"{fecha} {c.hora_fin_str}",
                'Resource': esp,
                'Color': color
            })
    
    # Gráfico de barras apiladas por especialidad y día
    esp_por_dia = defaultdict(lambda: defaultdict(int))
    for fecha, cirugias in cirugias_por_dia.items():
        for c in cirugias:
            esp = c.solicitud.tipo_intervencion.especialidad.name
            esp_por_dia[fecha.strftime('%a %d')][esp] += 1
    
    fig_barras = go.Figure()
    
    especialidades_usadas = set()
    for dia_data in esp_por_dia.values():
        especialidades_usadas.update(dia_data.keys())
    
    colores_esp = {
        'CIRUGIA_GENERAL': '#3498db',
        'CIRUGIA_DIGESTIVA': '#2ecc71',
        'CIRUGIA_COLORRECTAL': '#e74c3c',
        'UROLOGIA': '#f39c12',
        'GINECOLOGIA': '#9b59b6',
        'CIRUGIA_MAMA': '#e91e63',
        'CIRUGIA_VASCULAR': '#795548',
        'CIRUGIA_BARIATRICA': '#00bcd4',
        'CIRUGIA_ENDOCRINA': '#ff5722',
    }
    
    for esp in sorted(especialidades_usadas):
        valores = [esp_por_dia[fecha.strftime('%a %d')].get(esp, 0) for fecha in fechas_ordenadas]
        fig_barras.add_trace(go.Bar(
            name=ESPECIALIDADES_NOMBRES.get(esp, esp)[:10],
            x=[f.strftime('%a %d') for f in fechas_ordenadas],
            y=valores,
            marker_color=colores_esp.get(esp, '#95a5a6')
        ))
    
    fig_barras.update_layout(
        barmode='stack',
        title="Distribución por Especialidad y Día",
        xaxis_title="Día",
        yaxis_title="Nº Cirugías",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return md, fig_heatmap, fig_barras


# =============================================================================
# FUNCIONES PREDICTOR DE DEMANDA
# =============================================================================

def ejecutar_prediccion_demanda(semanas_horizonte, lista_espera_actual, fuera_plazo_actual):
    """Ejecuta la predicción de demanda y genera visualizaciones"""
    global predictor_demanda, PREDICTOR_DEMANDA_DISPONIBLE
    
    if not PREDICTOR_DEMANDA_DISPONIBLE or predictor_demanda is None:
        return ("⚠️ **Predictor de demanda no disponible.** "
                "Asegúrate de tener el archivo `predictor_demanda.py` en el mismo directorio.",
                None, None, pd.DataFrame())
    
    try:
        # Ejecutar predicción
        resultado = predictor_demanda.predecir(
            semanas=int(semanas_horizonte),
            lista_espera_actual=int(lista_espera_actual),
            fuera_plazo_actual=int(fuera_plazo_actual),
            n_simulaciones=100
        )
        
        # Generar markdown de resumen
        md = f"""
## 📈 Predicción de Demanda - Lista de Espera

**Fecha de predicción:** {resultado.fecha_prediccion}  
**Horizonte:** {resultado.semanas_horizonte} semanas  
**Modelo:** Series temporales + Monte Carlo (100 simulaciones)

### 📊 Calidad del Modelo
| Métrica | Valor |
|---------|-------|
| MAPE Entradas | {resultado.metricas_modelo.get('mape_entradas', 0):.1f}% |
| MAPE Salidas | {resultado.metricas_modelo.get('mape_salidas', 0):.1f}% |

### 📉 Estado Actual vs Proyección Final
| Métrica | Actual | Semana +{semanas_horizonte} | Cambio |
|---------|--------|------------|--------|
| Lista de espera | {resultado.lista_espera_actual} | {resultado.predicciones[-1].lista_espera_proyectada:.0f} | {'+' if resultado.predicciones[-1].lista_espera_proyectada > resultado.lista_espera_actual else ''}{resultado.predicciones[-1].lista_espera_proyectada - resultado.lista_espera_actual:.0f} |
| Fuera de plazo | {resultado.fuera_plazo_actual} | {resultado.predicciones[-1].fuera_plazo_proyectado:.0f} | {'+' if resultado.predicciones[-1].fuera_plazo_proyectado > resultado.fuera_plazo_actual else ''}{resultado.predicciones[-1].fuera_plazo_proyectado - resultado.fuera_plazo_actual:.0f} |

"""
        
        # Alertas
        if resultado.alertas:
            md += "### ⚠️ Alertas\n\n"
            for alerta in resultado.alertas:
                md += f"- {alerta}\n"
            md += "\n"
        
        # Recomendaciones
        if resultado.recomendaciones:
            md += "### 💡 Recomendaciones\n\n"
            for rec in resultado.recomendaciones:
                md += f"- {rec}\n"
            md += "\n"
        
        # Gráfico de evolución de lista de espera
        fig_evolucion = go.Figure()
        
        semanas = [0] + [p.semana for p in resultado.predicciones]
        lista_espera = [resultado.lista_espera_actual] + [p.lista_espera_proyectada for p in resultado.predicciones]
        fuera_plazo = [resultado.fuera_plazo_actual] + [p.fuera_plazo_proyectado for p in resultado.predicciones]
        
        # Banda de confianza (aproximada)
        lista_upper = [resultado.lista_espera_actual] + [
            p.lista_espera_proyectada + (p.entradas_ic_alto - p.entradas_media) * (i+1) * 0.5
            for i, p in enumerate(resultado.predicciones)
        ]
        lista_lower = [resultado.lista_espera_actual] + [
            max(0, p.lista_espera_proyectada - (p.entradas_media - p.entradas_ic_bajo) * (i+1) * 0.5)
            for i, p in enumerate(resultado.predicciones)
        ]
        
        # Banda de confianza
        fig_evolucion.add_trace(go.Scatter(
            x=semanas + semanas[::-1],
            y=lista_upper + lista_lower[::-1],
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='IC 80%',
            showlegend=True
        ))
        
        # Línea principal
        fig_evolucion.add_trace(go.Scatter(
            x=semanas, y=lista_espera,
            mode='lines+markers',
            name='Lista de Espera',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        
        # Fuera de plazo
        fig_evolucion.add_trace(go.Scatter(
            x=semanas, y=fuera_plazo,
            mode='lines+markers',
            name='Fuera de Plazo',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Línea de referencia actual
        fig_evolucion.add_hline(y=resultado.lista_espera_actual, line_dash="dot", 
                                line_color="gray", annotation_text="Actual")
        
        fig_evolucion.update_layout(
            title='📈 Proyección de Lista de Espera',
            xaxis_title='Semana',
            yaxis_title='Pacientes',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        # Gráfico de balance semanal
        fig_balance = go.Figure()
        
        entradas = [p.entradas_media for p in resultado.predicciones]
        salidas = [p.salidas_media for p in resultado.predicciones]
        balance = [p.balance_medio for p in resultado.predicciones]
        fechas = [p.fecha_inicio.strftime('%d/%m') for p in resultado.predicciones]
        
        fig_balance.add_trace(go.Bar(
            x=fechas, y=entradas,
            name='Entradas',
            marker_color='#e74c3c'
        ))
        
        fig_balance.add_trace(go.Bar(
            x=fechas, y=[-s for s in salidas],  # Negativo para mostrar abajo
            name='Salidas',
            marker_color='#27ae60'
        ))
        
        # Línea de balance
        fig_balance.add_trace(go.Scatter(
            x=fechas, y=balance,
            mode='lines+markers',
            name='Balance Neto',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig_balance.update_layout(
            title='📊 Balance Semanal: Entradas vs Salidas',
            xaxis_title='Semana',
            yaxis_title='Pacientes',
            yaxis2=dict(title='Balance', overlaying='y', side='right'),
            height=350,
            barmode='relative',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        # DataFrame con detalle
        datos_tabla = []
        for p in resultado.predicciones:
            datos_tabla.append({
                'Semana': f'+{p.semana}',
                'Fecha': p.fecha_inicio.strftime('%d/%m/%Y'),
                'Entradas': f"{p.entradas_media:.0f} [{p.entradas_ic_bajo:.0f}-{p.entradas_ic_alto:.0f}]",
                'Salidas': f"{p.salidas_media:.0f} [{p.salidas_ic_bajo:.0f}-{p.salidas_ic_alto:.0f}]",
                'Balance': f"{'+' if p.balance_medio >= 0 else ''}{p.balance_medio:.0f}",
                'Lista Espera': f"{p.lista_espera_proyectada:.0f}",
                'Fuera Plazo': f"{p.fuera_plazo_proyectado:.0f}"
            })
        
        df = pd.DataFrame(datos_tabla)
        
        return md, fig_evolucion, fig_balance, df
        
    except Exception as e:
        return f"❌ Error en predicción: {str(e)}", None, None, pd.DataFrame()


def obtener_stats_lista_espera():
    """Obtiene estadísticas actuales de la lista de espera"""
    if programador and programador.lista_espera:
        total = len(programador.lista_espera)
        fuera_plazo = sum(1 for s in programador.lista_espera if s.esta_fuera_plazo)
        return total, fuera_plazo
    return 500, 50


# =============================================================================
# FUNCIONES SIMULADOR WHAT-IF
# =============================================================================

def inicializar_simulador():
    """Inicializa el simulador con la configuración actual"""
    global simulador_whatif
    
    if not SIMULADOR_DISPONIBLE:
        return None
    
    # Calcular tasas de entrada desde histórico
    tasas_entrada = {}
    if programador and programador.historico is not None:
        hist = programador.historico
        semanas = hist['fecha'].nunique() / 5
        for esp in hist['especialidad'].unique():
            count = len(hist[hist['especialidad'] == esp])
            tasas_entrada[esp] = count / max(1, semanas)
    else:
        tasas_entrada = {
            'CIRUGIA_GENERAL': 15, 'CIRUGIA_DIGESTIVA': 10,
            'UROLOGIA': 12, 'GINECOLOGIA': 10, 'CIRUGIA_MAMA': 8,
            'CIRUGIA_COLORRECTAL': 6, 'CIRUGIA_VASCULAR': 7,
            'CIRUGIA_BARIATRICA': 4, 'CIRUGIA_ENDOCRINA': 4
        }
    
    # Obtener reservas del predictor si disponible
    reservas = {}
    if PREDICTOR_DISPONIBLE and predictor_urgencias:
        try:
            reservas_ml = predictor_urgencias.obtener_reservas_por_especialidad()
            reservas = {esp: datos['pct_reserva'] for esp, datos in reservas_ml.items()}
        except:
            pass
    
    lista_actual, fp_actual = obtener_stats_lista_espera()
    
    simulador_whatif = SimuladorWhatIf(
        configuracion_sesiones=configuracion_sesiones,
        lista_espera_actual=lista_actual,
        fuera_plazo_actual=fp_actual,
        tasas_entrada=tasas_entrada,
        reservas_urgencias=reservas
    )
    
    return simulador_whatif


def ejecutar_simulacion_whatif(tipo_escenario, especialidad, num_sesiones, 
                                quirofano_cerrar, factor_demanda, semanas):
    """Ejecuta una simulación what-if"""
    global simulador_whatif
    
    if not SIMULADOR_DISPONIBLE:
        return ("⚠️ Simulador no disponible. Asegúrate de tener `simulador_whatif.py`",
                None, None, pd.DataFrame())
    
    if simulador_whatif is None:
        simulador_whatif = inicializar_simulador()
        if simulador_whatif is None:
            return ("❌ Error inicializando simulador", None, None, pd.DataFrame())
    
    try:
        # Crear escenario según tipo
        if tipo_escenario == "Añadir sesiones":
            escenario = Escenario(
                nombre=f"+{int(num_sesiones)} sesiones {ESPECIALIDADES_NOMBRES.get(especialidad, especialidad)}",
                tipo=TipoEscenario.AÑADIR_SESIONES,
                sesiones_extra={especialidad: int(num_sesiones)},
                semanas_duracion=int(semanas)
            )
        elif tipo_escenario == "Quitar sesiones":
            escenario = Escenario(
                nombre=f"-{int(num_sesiones)} sesiones {ESPECIALIDADES_NOMBRES.get(especialidad, especialidad)}",
                tipo=TipoEscenario.AÑADIR_SESIONES,
                sesiones_extra={especialidad: -int(num_sesiones)},
                semanas_duracion=int(semanas)
            )
        elif tipo_escenario == "Cerrar quirófano":
            escenario = Escenario(
                nombre=f"Cerrar Q{int(quirofano_cerrar)}",
                tipo=TipoEscenario.CERRAR_QUIROFANO,
                dias_cierre=int(semanas) * 7,
                semanas_duracion=int(semanas)
            )
        elif tipo_escenario == "Cambio demanda":
            escenario = Escenario(
                nombre=f"Demanda x{factor_demanda:.0%}",
                tipo=TipoEscenario.AUMENTAR_DEMANDA if factor_demanda > 1 else TipoEscenario.REDUCIR_DEMANDA,
                factor_demanda=factor_demanda,
                semanas_duracion=int(semanas)
            )
        else:
            return ("❌ Tipo de escenario no reconocido", None, None, pd.DataFrame())
        
        # Ejecutar simulación
        resultado = simulador_whatif.simular_escenario(escenario, n_simulaciones=300)
        
        lista_actual, fp_actual = obtener_stats_lista_espera()
        
        md = f"""
## 🔮 Simulación What-If: {escenario.nombre}

**Horizonte:** {semanas} semanas | **Simulaciones:** 300 (Monte Carlo)

### 📊 Comparación con Situación Actual

| Métrica | Actual | Proyección | Cambio |
|---------|--------|------------|--------|
| Lista de espera | {lista_actual} | {resultado.lista_final_media:.0f} | {resultado.diferencia_lista:+.0f} |
| Fuera de plazo | {fp_actual} | {resultado.fp_final_media:.0f} | {resultado.diferencia_fp:+.0f} |

### 📈 Intervalos de Confianza (80%)

| Métrica | Optimista (P10) | Esperado | Pesimista (P90) |
|---------|-----------------|----------|-----------------|
| Lista final | {resultado.lista_final_p10:.0f} | {resultado.lista_final_media:.0f} | {resultado.lista_final_p90:.0f} |

### 🎯 Probabilidades

- Probabilidad de reducir lista: **{resultado.prob_reducir_lista:.0%}**
- Utilización del sistema: **{resultado.utilizacion_sistema:.0%}**
"""
        
        if resultado.recomendaciones:
            md += "\n### 💡 Recomendaciones\n"
            for rec in resultado.recomendaciones:
                md += f"- {rec}\n"
        
        # Gráfico de evolución
        fig_evolucion = go.Figure()
        semanas_x = list(range(int(semanas) + 1))
        
        fig_evolucion.add_trace(go.Scatter(
            x=semanas_x + semanas_x[::-1],
            y=resultado.lista_ic_alto + resultado.lista_ic_bajo[::-1],
            fill='toself', fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='IC 80%'
        ))
        
        fig_evolucion.add_trace(go.Scatter(
            x=semanas_x, y=resultado.lista_espera,
            mode='lines+markers', name='Lista de Espera',
            line=dict(color='#3498db', width=3)
        ))
        
        fig_evolucion.add_trace(go.Scatter(
            x=semanas_x, y=resultado.fuera_plazo,
            mode='lines+markers', name='Fuera de Plazo',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        fig_evolucion.add_hline(y=lista_actual, line_dash="dot", line_color="gray")
        fig_evolucion.update_layout(title=f'📈 Proyección: {escenario.nombre}',
                                    xaxis_title='Semana', yaxis_title='Pacientes', height=400)
        
        # Gráfico de barras
        fig_barras = go.Figure()
        fig_barras.add_trace(go.Bar(name='Actual', x=['Lista', 'FP'], 
                                    y=[lista_actual, fp_actual], marker_color='#95a5a6'))
        fig_barras.add_trace(go.Bar(name='Proyección', x=['Lista', 'FP'],
                                    y=[resultado.lista_final_media, resultado.fp_final_media],
                                    marker_color='#3498db'))
        fig_barras.update_layout(title='📊 Actual vs Proyección', barmode='group', height=350)
        
        # Tabla evolución
        datos_tabla = []
        for i in range(0, min(len(semanas_x), len(resultado.lista_espera)), 2):
            datos_tabla.append({
                'Semana': f'+{i}',
                'Lista': f"{resultado.lista_espera[i]:.0f}",
                'FP': f"{resultado.fuera_plazo[i]:.0f}" if i < len(resultado.fuera_plazo) else "-",
                'Δ Lista': f"{resultado.lista_espera[i] - lista_actual:+.0f}"
            })
        
        return md, fig_evolucion, fig_barras, pd.DataFrame(datos_tabla)
        
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}", None, None, pd.DataFrame()


def comparar_escenarios_whatif(especialidad, semanas):
    """Compara múltiples escenarios"""
    global simulador_whatif
    
    if not SIMULADOR_DISPONIBLE:
        return "⚠️ Simulador no disponible", None, pd.DataFrame()
    
    if simulador_whatif is None:
        simulador_whatif = inicializar_simulador()
    
    try:
        escenarios = [
            Escenario(nombre="Baseline", tipo=TipoEscenario.PERSONALIZADO, semanas_duracion=int(semanas)),
            Escenario(nombre=f"+1 sesión", tipo=TipoEscenario.AÑADIR_SESIONES,
                     sesiones_extra={especialidad: 1}, semanas_duracion=int(semanas)),
            Escenario(nombre=f"+2 sesiones", tipo=TipoEscenario.AÑADIR_SESIONES,
                     sesiones_extra={especialidad: 2}, semanas_duracion=int(semanas)),
            Escenario(nombre=f"+3 sesiones", tipo=TipoEscenario.AÑADIR_SESIONES,
                     sesiones_extra={especialidad: 3}, semanas_duracion=int(semanas)),
        ]
        
        resultados = []
        for esc in escenarios:
            res = simulador_whatif.simular_escenario(esc, 200)
            resultados.append({
                'escenario': esc.nombre,
                'lista_final': res.lista_final_media,
                'fp_final': res.fp_final_media,
                'diff_lista': res.diferencia_lista
            })
        
        df = pd.DataFrame([{
            'Escenario': r['escenario'],
            'Lista Final': f"{r['lista_final']:.0f}",
            'Fuera Plazo': f"{r['fp_final']:.0f}",
            'Δ Lista': f"{r['diff_lista']:+.0f}"
        } for r in resultados])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Lista', x=[r['escenario'] for r in resultados],
                            y=[r['lista_final'] for r in resultados], marker_color='#3498db'))
        fig.update_layout(title=f'📊 Comparativa: {ESPECIALIDADES_NOMBRES.get(especialidad, especialidad)}',
                         height=400)
        
        mejor = min(resultados, key=lambda x: x['lista_final'])
        md = f"""
## 📊 Comparativa de Escenarios

**Especialidad:** {ESPECIALIDADES_NOMBRES.get(especialidad, especialidad)} | **Horizonte:** {semanas} semanas

### 🏆 Mejor: {mejor['escenario']}
- Lista final: **{mejor['lista_final']:.0f}** pacientes
- Reducción: **{abs(mejor['diff_lista']):.0f}** pacientes
"""
        return md, fig, df
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, pd.DataFrame()


def calcular_sesiones_necesarias_ui(especialidad, semanas):
    """Calcula sesiones necesarias para eliminar FP"""
    global simulador_whatif
    
    if not SIMULADOR_DISPONIBLE:
        return "⚠️ Simulador no disponible"
    
    if simulador_whatif is None:
        simulador_whatif = inicializar_simulador()
    
    try:
        resultado = simulador_whatif.optimizador.encontrar_sesiones_minimas(
            especialidad, objetivo_fp=10, semanas=int(semanas)
        )
        
        return f"""
## 🎯 Sesiones Necesarias

**Especialidad:** {ESPECIALIDADES_NOMBRES.get(especialidad, especialidad)}  
**Plazo:** {semanas} semanas

| Métrica | Valor |
|---------|-------|
| **Sesiones necesarias** | **+{resultado['sesiones_necesarias']}**/semana |
| Lista final esperada | {resultado['lista_proyectada']:.0f} |
| FP final esperado | {resultado['fp_proyectado']:.0f} |
"""
    except Exception as e:
        return f"❌ Error: {str(e)}"


# =============================================================================
# PLANIFICADOR ESTRATÉGICO INTEGRADO
# =============================================================================

def ejecutar_planificacion_estrategica(horizonte_semanas):
    """
    Función integrada que:
    1. Analiza la demanda actual
    2. Calcula el reparto óptimo de sesiones
    3. Simula el impacto con What-If
    """
    global simulador_whatif, configuracion_optima_calculada
    
    try:
        demanda = calcular_demanda_por_especialidad()
        if not demanda:
            return "❌ No hay datos de demanda", None, None, pd.DataFrame(), ""
        
        lista_actual, fp_actual = obtener_stats_lista_espera()
        
        # Calcular configuración óptima
        config_optima = calcular_configuracion_optima()
        if not config_optima:
            return "❌ Error calculando config óptima", None, None, pd.DataFrame(), ""
        
        # Comparar sesiones
        sesiones_actuales = {}
        sesiones_optimas = {}
        
        for esp in LISTA_ESPECIALIDADES:
            if esp in ['LIBRE', 'CERRADO']:
                continue
            
            count_actual = sum(1 for q in configuracion_sesiones for dia in DIAS_SEMANA 
                              for turno in ['Mañana', 'Tarde']
                              if configuracion_sesiones[q][dia][turno] == esp)
            count_optimo = sum(1 for q in config_optima for dia in DIAS_SEMANA
                              for turno in ['Mañana', 'Tarde']
                              if config_optima[q][dia][turno] == esp)
            
            if count_actual > 0 or count_optimo > 0:
                sesiones_actuales[esp] = count_actual
                sesiones_optimas[esp] = count_optimo
        
        diferencias = {esp: sesiones_optimas.get(esp, 0) - sesiones_actuales.get(esp, 0)
                      for esp in set(sesiones_actuales) | set(sesiones_optimas)}
        
        # Simular con What-If
        proyeccion_actual = None
        proyeccion_optima = None
        
        if SIMULADOR_DISPONIBLE:
            if simulador_whatif is None:
                simulador_whatif = inicializar_simulador()
            
            if simulador_whatif:
                esc_actual = Escenario(nombre="Actual", tipo=TipoEscenario.PERSONALIZADO,
                                       semanas_duracion=int(horizonte_semanas))
                proyeccion_actual = simulador_whatif.simular_escenario(esc_actual, 200)
                
                sesiones_extra = {esp: d for esp, d in diferencias.items() if d > 0}
                if sesiones_extra:
                    esc_optimo = Escenario(nombre="Óptimo", tipo=TipoEscenario.AÑADIR_SESIONES,
                                          sesiones_extra=sesiones_extra,
                                          semanas_duracion=int(horizonte_semanas))
                    proyeccion_optima = simulador_whatif.simular_escenario(esc_optimo, 200)
        
        # Generar informe
        md = f"""
## 🎯 Planificación Estratégica

### 📊 Estado Actual
| Métrica | Valor |
|---------|-------|
| Pacientes en lista | **{lista_actual}** |
| Fuera de plazo | **{fp_actual}** ({fp_actual/max(1,lista_actual)*100:.1f}%) |

### 🔄 Sesiones: Actual vs Óptimo

| Especialidad | Actual | Óptimo | Δ |
|--------------|--------|--------|---|
"""
        for esp in sorted(diferencias.keys(), key=lambda x: abs(diferencias[x]), reverse=True):
            md += f"| {ESPECIALIDADES_NOMBRES.get(esp, esp)[:15]} | {sesiones_actuales.get(esp,0)} | {sesiones_optimas.get(esp,0)} | {diferencias[esp]:+d} |\n"
        
        if proyeccion_actual and proyeccion_optima:
            md += f"""

### 📈 Proyección a {horizonte_semanas} semanas

| Escenario | Lista Final | FP Final | Prob. Reducir |
|-----------|-------------|----------|---------------|
| Mantener actual | {proyeccion_actual.lista_final_media:.0f} | {proyeccion_actual.fp_final_media:.0f} | {proyeccion_actual.prob_reducir_lista:.0%} |
| Aplicar óptimo | {proyeccion_optima.lista_final_media:.0f} | {proyeccion_optima.fp_final_media:.0f} | {proyeccion_optima.prob_reducir_lista:.0%} |
"""
        
        # Gráficos
        fig_sesiones = go.Figure()
        nombres = [ESPECIALIDADES_NOMBRES.get(e, e)[:10] for e in diferencias.keys()]
        fig_sesiones.add_trace(go.Bar(name='Actual', x=nombres, 
                                      y=[sesiones_actuales.get(e,0) for e in diferencias.keys()],
                                      marker_color='#95a5a6'))
        fig_sesiones.add_trace(go.Bar(name='Óptimo', x=nombres,
                                      y=[sesiones_optimas.get(e,0) for e in diferencias.keys()],
                                      marker_color='#3498db'))
        fig_sesiones.update_layout(title='Sesiones: Actual vs Óptimo', barmode='group', height=400)
        
        fig_proyeccion = None
        if proyeccion_actual and proyeccion_optima:
            fig_proyeccion = go.Figure()
            semanas_x = list(range(int(horizonte_semanas) + 1))
            # Usar lista_espera en lugar de lista_espera_proyectada
            y_actual = proyeccion_actual.lista_espera[:len(semanas_x)] if len(proyeccion_actual.lista_espera) >= len(semanas_x) else proyeccion_actual.lista_espera
            y_optimo = proyeccion_optima.lista_espera[:len(semanas_x)] if len(proyeccion_optima.lista_espera) >= len(semanas_x) else proyeccion_optima.lista_espera
            fig_proyeccion.add_trace(go.Scatter(x=semanas_x[:len(y_actual)], y=y_actual,
                                               mode='lines', name='Actual', line=dict(color='#95a5a6', dash='dash')))
            fig_proyeccion.add_trace(go.Scatter(x=semanas_x[:len(y_optimo)], y=y_optimo,
                                               mode='lines+markers', name='Óptimo', line=dict(color='#3498db', width=3)))
            fig_proyeccion.add_hline(y=lista_actual, line_dash="dot", line_color="gray")
            fig_proyeccion.update_layout(title='Proyección Lista de Espera', height=400)
        
        # Tabla demanda
        datos_demanda = [{'Especialidad': ESPECIALIDADES_NOMBRES.get(esp, esp),
                         'Pacientes': d['cantidad'], 'Oncológicos': d.get('oncologicos', 0),
                         'Fuera Plazo': d.get('fuera_plazo', 0)}
                        for esp, d in demanda.items()]
        
        return md, fig_sesiones, fig_proyeccion, pd.DataFrame(datos_demanda), "✅ Usa el botón para aplicar"
        
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}", None, None, pd.DataFrame(), ""


# =============================================================================
# FUNCIONES DE UI PARA TABS NUEVOS
# =============================================================================

def generar_dashboard():
    """Genera el dashboard principal con métricas del sistema"""
    try:
        stats = obtener_stats_lista_espera()
        total, fp = stats
        
        # Usar la lista de espera del programador
        le = programador.lista_espera if programador and programador.lista_espera else []
        
        if not le:
            return "⚠️ No hay datos de lista de espera", None
        
        # Estadísticas por prioridad
        prioridades = {}
        for s in le:
            p = s.prioridad.name
            prioridades[p] = prioridades.get(p, 0) + 1
        
        # Estadísticas por especialidad
        especialidades = {}
        for s in le:
            e = s.tipo_intervencion.especialidad.name
            especialidades[e] = especialidades.get(e, 0) + 1
        
        md = f"""
## 📊 Estado del Bloque Quirúrgico

### Resumen Lista de Espera
| Métrica | Valor |
|---------|-------|
| **Total pacientes** | {total} |
| **Fuera de plazo** | {fp} ({fp/total*100:.1f}%) |
| **Oncológicos** | {sum(1 for s in le if 'ONCOLOG' in s.prioridad.name)} |

### Por Prioridad
| Prioridad | Pacientes |
|-----------|-----------|
"""
        for p, count in sorted(prioridades.items(), key=lambda x: -x[1])[:6]:
            md += f"| {p} | {count} |\n"
        
        md += """
### Por Especialidad (Top 6)
| Especialidad | Pacientes |
|--------------|-----------|
"""
        for e, count in sorted(especialidades.items(), key=lambda x: -x[1])[:6]:
            md += f"| {e.replace('CIRUGIA_', '')} | {count} |\n"
        
        # Gráfico
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(especialidades.keys()),
            y=list(especialidades.values()),
            marker_color='steelblue'
        ))
        fig.update_layout(
            title="Pacientes por Especialidad",
            xaxis_tickangle=-45,
            height=400
        )
        
        return md, fig
    except Exception as e:
        return f"❌ Error generando dashboard: {str(e)}", None


def mostrar_lista_espera():
    """Muestra la lista de espera en formato DataFrame"""
    try:
        le = programador.lista_espera if programador and programador.lista_espera else []
        if not le:
            return pd.DataFrame({'Mensaje': ['No hay datos de lista de espera']})
        
        datos = []
        for s in le[:100]:  # Limitar a 100 para rendimiento
            datos.append({
                'ID': s.id,
                'Paciente': s.paciente.nombre[:30],
                'Intervención': s.tipo_intervencion.nombre[:25],
                'Prioridad': s.prioridad.name.replace('_', ' '),
                'Días espera': s.dias_en_espera,
                'Fuera plazo': '⚠️ SÍ' if s.esta_fuera_plazo else 'No',
                'Score': round(s.score_clinico, 1)
            })
        return pd.DataFrame(datos)
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})


def mostrar_prediccion_urgencias():
    """Muestra predicción de urgencias por especialidad"""
    try:
        if urgencias_predictor is None:
            return "❌ Predictor no disponible", pd.DataFrame(), None
        
        reservas = urgencias_predictor.obtener_reservas_por_especialidad()
        
        md = """
## 🔮 Predicción de Reservas para Urgencias

El modelo ML analiza el histórico para predecir cuánto tiempo reservar 
en cada sesión para urgencias diferidas.
"""
        
        datos = []
        for esp, info in reservas.items():
            datos.append({
                'Especialidad': esp.replace('CIRUGIA_', ''),
                'Tasa urgencias': f"{info.get('tasa_urgencias', 0)*100:.1f}%",
                'Reserva mañana': f"{info.get('minutos_reserva_manana', 0):.0f} min",
                'Reserva tarde': f"{info.get('minutos_reserva_tarde', 0):.0f} min",
                'Confianza': f"{info.get('confianza', 0)*100:.0f}%"
            })
        
        df = pd.DataFrame(datos)
        
        # Gráfico
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[d['Especialidad'] for d in datos],
            y=[info.get('minutos_reserva_manana', 0) for info in reservas.values()],
            name='Mañana',
            marker_color='coral'
        ))
        fig.add_trace(go.Bar(
            x=[d['Especialidad'] for d in datos],
            y=[info.get('minutos_reserva_tarde', 0) for info in reservas.values()],
            name='Tarde',
            marker_color='lightsalmon'
        ))
        fig.update_layout(
            title="Reserva sugerida por especialidad (minutos)", 
            height=350,
            barmode='group'
        )
        
        return md, df, fig
    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame(), None


def mostrar_prediccion_semanal():
    """Muestra predicción semanal de urgencias"""
    try:
        if urgencias_predictor is None:
            return "❌ Predictor no disponible", pd.DataFrame()
        
        from datetime import date, timedelta
        
        md = "## 📅 Predicción para próximos 5 días laborables\n\n"
        datos = []
        
        fecha = date.today() + timedelta(days=1)
        dias_procesados = 0
        
        while dias_procesados < 5:
            if fecha.weekday() < 5:  # Solo días laborables
                for esp in urgencias_predictor.modelos.keys():
                    try:
                        pred = urgencias_predictor.predecir(esp, fecha)
                        datos.append({
                            'Fecha': fecha.strftime('%a %d/%m'),
                            'Especialidad': esp.replace('CIRUGIA_', ''),
                            'Urgencias esperadas': f"{pred.urgencias_esperadas:.1f}",
                            'Reserva mañana': f"{int(pred.pct_reserva_recomendado * 420 / 100)} min",
                            'Confianza': f"{pred.confianza*100:.0f}%"
                        })
                    except:
                        pass
                dias_procesados += 1
            fecha += timedelta(days=1)
        
        return md, pd.DataFrame(datos)
    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame()


def aplicar_reservas_ml():
    """Aplica las reservas ML al optimizador"""
    if urgencias_predictor is None:
        return "❌ Predictor no disponible"
    
    reservas = urgencias_predictor.obtener_reservas_por_especialidad()
    
    md = """
## ✅ Reservas ML Aplicadas

Las siguientes reservas se usarán en la próxima optimización:

| Especialidad | Reserva Mañana | Reserva Tarde |
|--------------|----------------|---------------|
"""
    for esp, info in reservas.items():
        md += f"| {esp.replace('CIRUGIA_', '')} | {info.get('minutos_reserva_manana', 0)} min | {info.get('minutos_reserva_tarde', 0)} min |\n"
    
    md += "\n**Nota:** Ejecuta 'Optimizar' para ver el efecto de estas reservas."
    
    return md


# =============================================================================
# INTERFAZ GRADIO
# =============================================================================

print("\n🚀 Construyendo interfaz...")

with gr.Blocks(title="Programador Quirúrgico v4.9") as demo:
    gr.Markdown("""
    # 🏥 Programador Quirúrgico v4.9
    ### Flujo: 📊 Análisis → 🎯 Planificación → ⚙️ Configuración → 📅 Ejecución
    """)
    
    with gr.Tabs():
        
        # =====================================================================
        # FASE 1: ANÁLISIS - ¿Qué tenemos?
        # =====================================================================
        
        # TAB DASHBOARD (movido al inicio)
        with gr.TabItem("📊 Dashboard"):
            gr.Markdown("### Vista general del estado del bloque quirúrgico")
            btn_dash = gr.Button("🔄 Actualizar Dashboard", variant="primary")
            dash_md = gr.Markdown()
            dash_fig = gr.Plot()
            btn_dash.click(generar_dashboard, outputs=[dash_md, dash_fig])
        
        # TAB LISTA DE ESPERA
        with gr.TabItem("📋 Lista Espera"):
            gr.Markdown("### Pacientes en lista de espera quirúrgica")
            btn_lista = gr.Button("🔄 Actualizar Lista", variant="primary")
            lista_df = gr.Dataframe()
            btn_lista.click(mostrar_lista_espera, outputs=[lista_df])
        
        # TAB PREDICCIÓN DEMANDA
        with gr.TabItem("📈 Pred. Demanda"):
            gr.Markdown("""
            ### 📈 Predicción de Movimientos de Lista de Espera
            
            Analiza cómo evolucionará la lista basándose en entradas y salidas históricas.
            """)
            
            with gr.Row():
                pred_semanas = gr.Slider(4, 24, 12, step=2, label="Semanas a predecir")
                pred_lista = gr.Number(value=500, label="Lista actual (override)")
                pred_fp = gr.Number(value=50, label="Fuera plazo actual")
            
            btn_pred_dem = gr.Button("📈 Ejecutar Predicción", variant="primary", size="lg")
            
            pred_dem_md = gr.Markdown()
            with gr.Row():
                pred_dem_fig1 = gr.Plot()
                pred_dem_fig2 = gr.Plot()
            pred_dem_tabla = gr.Dataframe()
            
            btn_pred_dem.click(ejecutar_prediccion_demanda,
                              inputs=[pred_semanas, pred_lista, pred_fp],
                              outputs=[pred_dem_md, pred_dem_fig1, pred_dem_fig2, pred_dem_tabla])
        
        # =====================================================================
        # FASE 2: PLANIFICACIÓN ESTRATÉGICA - ¿Qué debería hacer?
        # =====================================================================
        
        # TAB PLANIFICADOR ESTRATÉGICO (NUEVO)
        with gr.TabItem("🎯 Planificador"):
            gr.Markdown("""
            ### 🎯 Planificador Estratégico
            
            **Análisis integral:**
            1. 📊 Demanda actual por especialidad
            2. 🔄 Reparto óptimo de sesiones
            3. 🔮 Simulación What-If con Monte Carlo
            4. 💡 Recomendaciones concretas
            """)
            
            plan_horizonte = gr.Slider(4, 24, 12, step=2, label="Horizonte (semanas)")
            btn_planificar = gr.Button("🎯 Ejecutar Planificación Estratégica", variant="primary", size="lg")
            
            plan_md = gr.Markdown()
            with gr.Row():
                plan_fig_sesiones = gr.Plot()
                plan_fig_proyeccion = gr.Plot()
            
            plan_tabla = gr.Dataframe()
            plan_msg = gr.Markdown()
            btn_aplicar_plan = gr.Button("✅ Aplicar Configuración Óptima", variant="secondary")
            
            def aplicar_config_simple():
                """Aplica la configuración óptima y retorna solo mensaje"""
                global configuracion_optima_calculada
                if configuracion_optima_calculada is None:
                    return "⚠️ Primero ejecuta la planificación estratégica"
                
                # Aplicar configuración
                for q, datos in configuracion_optima_calculada.items():
                    if isinstance(q, int) and q in configuracion_sesiones:
                        for dia in DIAS_SEMANA:
                            if dia in datos:
                                configuracion_sesiones[q][dia] = datos[dia].copy()
                
                return "✅ **Configuración óptima aplicada.** Ve a la pestaña '🗓️ Sesiones' para ver los cambios."
            
            btn_planificar.click(ejecutar_planificacion_estrategica,
                                inputs=[plan_horizonte],
                                outputs=[plan_md, plan_fig_sesiones, plan_fig_proyeccion, plan_tabla, plan_msg])
            btn_aplicar_plan.click(aplicar_config_simple, outputs=[plan_msg])
        
        # TAB SIMULADOR WHAT-IF
        with gr.TabItem("🔮 What-If"):
            gr.Markdown("""
            ### 🔮 Simulador de Escenarios What-If
            
            Simula: *¿Qué pasa si añado sesiones? ¿Si cierro un quirófano?*
            """)
            
            with gr.Tabs():
                with gr.TabItem("📊 Simular"):
                    with gr.Row():
                        with gr.Column():
                            tipo_esc = gr.Dropdown(choices=["Añadir sesiones", "Quitar sesiones", 
                                                           "Cerrar quirófano", "Cambio demanda"],
                                                  value="Añadir sesiones", label="Tipo")
                            esp_sim = gr.Dropdown(choices=[e for e in LISTA_ESPECIALIDADES if e not in ['LIBRE','CERRADO']],
                                                 value='CIRUGIA_DIGESTIVA', label="Especialidad")
                            n_ses = gr.Slider(1, 5, 2, step=1, label="Nº sesiones")
                        with gr.Column():
                            q_cerrar = gr.Slider(1, 8, 3, step=1, label="Quirófano a cerrar")
                            f_dem = gr.Slider(0.7, 1.5, 1.0, step=0.05, label="Factor demanda")
                            sem_sim = gr.Slider(4, 24, 12, step=2, label="Semanas")
                    
                    btn_sim = gr.Button("🔮 Simular", variant="primary", size="lg")
                    sim_md = gr.Markdown()
                    with gr.Row():
                        sim_fig1 = gr.Plot()
                        sim_fig2 = gr.Plot()
                    sim_tabla = gr.Dataframe()
                    
                    btn_sim.click(ejecutar_simulacion_whatif,
                                 inputs=[tipo_esc, esp_sim, n_ses, q_cerrar, f_dem, sem_sim],
                                 outputs=[sim_md, sim_fig1, sim_fig2, sim_tabla])
                
                with gr.TabItem("📈 Comparar"):
                    with gr.Row():
                        esp_comp = gr.Dropdown(choices=[e for e in LISTA_ESPECIALIDADES if e not in ['LIBRE','CERRADO']],
                                              value='CIRUGIA_DIGESTIVA', label="Especialidad")
                        sem_comp = gr.Slider(4, 24, 12, step=2, label="Semanas")
                    btn_comp = gr.Button("📊 Comparar", variant="primary")
                    comp_md = gr.Markdown()
                    comp_fig = gr.Plot()
                    comp_tabla = gr.Dataframe()
                    btn_comp.click(comparar_escenarios_whatif, inputs=[esp_comp, sem_comp],
                                  outputs=[comp_md, comp_fig, comp_tabla])
                
                with gr.TabItem("🎯 Calculadora"):
                    gr.Markdown("### ¿Cuántas sesiones necesito para eliminar FP?")
                    with gr.Row():
                        esp_calc = gr.Dropdown(choices=[e for e in LISTA_ESPECIALIDADES if e not in ['LIBRE','CERRADO']],
                                              value='CIRUGIA_GENERAL', label="Especialidad")
                        sem_calc = gr.Slider(4, 24, 12, step=2, label="Plazo (semanas)")
                    btn_calc = gr.Button("🎯 Calcular", variant="primary")
                    calc_md = gr.Markdown()
                    btn_calc.click(calcular_sesiones_necesarias_ui, inputs=[esp_calc, sem_calc], outputs=[calc_md])
        
        # TAB PREDICCIÓN URGENCIAS
        with gr.TabItem("🚑 Pred. Urgencias"):
            gr.Markdown("""
            ### 🔮 Predicción de Urgencias Diferidas
            
            ML para predecir cuánto tiempo reservar en cada sesión para urgencias.
            """)
            with gr.Tabs():
                with gr.TabItem("📊 Resumen"):
                    btn_pred_urg = gr.Button("🔄 Calcular Predicción", variant="primary")
                    pred_urg_md = gr.Markdown()
                    pred_urg_df = gr.Dataframe()
                    pred_urg_fig = gr.Plot()
                    btn_pred_urg.click(mostrar_prediccion_urgencias, outputs=[pred_urg_md, pred_urg_df, pred_urg_fig])
                
                with gr.TabItem("📅 Semanal"):
                    btn_pred_sem = gr.Button("📅 Predicción Semanal", variant="primary")
                    pred_sem_md = gr.Markdown()
                    pred_sem_df = gr.Dataframe()
                    btn_pred_sem.click(mostrar_prediccion_semanal, outputs=[pred_sem_md, pred_sem_df])
                
                with gr.TabItem("⚙️ Aplicar"):
                    gr.Markdown("Aplica las reservas calculadas por ML al optimizador")
                    btn_aplicar_res = gr.Button("✅ Aplicar Reservas ML", variant="primary", size="lg")
                    aplicar_res_md = gr.Markdown()
                    btn_aplicar_res.click(aplicar_reservas_ml, outputs=[aplicar_res_md])
        
        # =====================================================================
        # FASE 3: CONFIGURACIÓN TÁCTICA - ¿Cómo lo configuro?
        # =====================================================================
        
        # TAB SESIONES
        with gr.TabItem("🗓️ Sesiones"):
            gr.Markdown("**Mañana:** 08:00-15:00 | **Tarde:** 15:00-20:00")
            
            dropdowns = []
            for q in range(1, NUM_QUIROFANOS + 1):
                with gr.Accordion(f"Quirófano {q}", open=(q <= 2)):
                    with gr.Row():
                        for dia in DIAS_SEMANA:
                            with gr.Column(min_width=90):
                                gr.Markdown(f"**{dia[:3]}**")
                                dd_m = gr.Dropdown(LISTA_ESPECIALIDADES, value=configuracion_sesiones[q][dia]['Mañana'], label="M")
                                dropdowns.append(dd_m)
                                dd_t = gr.Dropdown(LISTA_ESPECIALIDADES, value=configuracion_sesiones[q][dia]['Tarde'], label="T")
                                dropdowns.append(dd_t)
            
            with gr.Row():
                btn_guardar = gr.Button("💾 Guardar", variant="primary")
                btn_reset = gr.Button("🔄 Reset")
            
            cfg_msg = gr.Markdown()
            cfg_res = gr.Markdown(value=generar_resumen_configuracion())
            cfg_mat = gr.Dataframe(value=generar_matriz_visual())
            cfg_fig = gr.Plot(value=generar_grafico_sesiones())
            
            btn_guardar.click(guardar_configuracion, inputs=dropdowns, outputs=[cfg_msg, cfg_res, cfg_mat, cfg_fig])
            btn_reset.click(resetear_configuracion, outputs=[cfg_msg, cfg_res, cfg_mat, cfg_fig] + dropdowns)
        
        # TAB RESTRICCIONES
        with gr.TabItem("🚫 Restricciones"):
            gr.Markdown("""
            ### Constructor de Restricciones
            
            Define restricciones que el sistema debe respetar al calcular la configuración óptima.
            Selecciona un tipo de restricción y completa los parámetros necesarios.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Selector de tipo de restricción
                    tipo_restr = gr.Dropdown(
                        choices=[(CATALOGO_RESTRICCIONES[k]['nombre'], k) for k in CATALOGO_RESTRICCIONES.keys()],
                        label="Tipo de restricción",
                        info="Selecciona qué tipo de restricción quieres añadir"
                    )
                    
                    # Descripción del tipo seleccionado
                    desc_tipo = gr.Markdown("*Selecciona un tipo de restricción para ver su descripción*")
                
                with gr.Column(scale=3):
                    gr.Markdown("**Parámetros:**")
                    with gr.Row():
                        param_esp = gr.Dropdown(
                            choices=[(ESPECIALIDADES_NOMBRES.get(e, e), e) for e in ESPECIALIDADES_SIN_LIBRE],
                            label="Especialidad",
                            visible=False
                        )
                        param_dia = gr.Dropdown(
                            choices=DIAS_SEMANA,
                            label="Día",
                            visible=False
                        )
                        param_turno = gr.Dropdown(
                            choices=OPCIONES_TURNO,
                            label="Turno",
                            visible=False
                        )
                    with gr.Row():
                        param_quirofano = gr.Dropdown(
                            choices=OPCIONES_QUIROFANO,
                            label="Quirófano",
                            visible=False
                        )
                        param_cantidad = gr.Dropdown(
                            choices=OPCIONES_CANTIDAD,
                            label="Cantidad",
                            visible=False
                        )
            
            with gr.Row():
                btn_añadir = gr.Button("➕ Añadir restricción", variant="primary")
                msg_restriccion = gr.Markdown()
            
            gr.Markdown("---")
            
            # Lista de restricciones activas
            lista_restricciones = gr.Markdown(value=actualizar_lista_restricciones())
            
            with gr.Row():
                input_eliminar = gr.Number(label="Nº a eliminar", precision=0, minimum=1, maximum=20)
                btn_eliminar = gr.Button("🗑️ Eliminar", variant="secondary")
                btn_limpiar = gr.Button("🧹 Limpiar todas", variant="stop")
            
            msg_eliminar = gr.Markdown()
            
            # Ejemplos de uso
            with gr.Accordion("💡 Ejemplos de restricciones", open=False):
                gr.Markdown("""
                | Situación | Tipo de restricción | Parámetros |
                |-----------|---------------------|------------|
                | El servicio de urología no opera los miércoles | Cerrar especialidad un día | Urología, Miércoles, Ambos |
                | Q3 en mantenimiento el viernes tarde | Cerrar quirófano | 3, Viernes, Tarde |
                | Cirugía bariátrica solo tiene equipo de mañana | Especialidad solo mañanas | Bariátrica |
                | Máximo 5 sesiones semanales de mama | Máximo sesiones por especialidad | Mama, 5 |
                | Digestivo debe tener al menos 6 sesiones | Mínimo sesiones por especialidad | Digestivo, 6 |
                | Vascular solo puede operar en Q7 | Especialidad en quirófano fijo | Vascular, 7 |
                | Festivo el lunes | Cerrar día completo | Lunes |
                | No hay anestesistas para tardes el viernes | Cerrar turno en un día | Tarde, Viernes |
                """)
            
            # Función para actualizar descripción y visibilidad de parámetros
            def actualizar_formulario(tipo):
                if not tipo or tipo not in CATALOGO_RESTRICCIONES:
                    return (
                        "*Selecciona un tipo de restricción para ver su descripción*",
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False)
                    )
                
                info = CATALOGO_RESTRICCIONES[tipo]
                desc = f"**{info['nombre']}**: {info['descripcion']}"
                params = info['parametros']
                
                return (
                    desc,
                    gr.update(visible='especialidad' in params),
                    gr.update(visible='dia' in params),
                    gr.update(visible='turno' in params),
                    gr.update(visible='quirofano' in params),
                    gr.update(visible='cantidad' in params)
                )
            
            tipo_restr.change(
                actualizar_formulario,
                inputs=[tipo_restr],
                outputs=[desc_tipo, param_esp, param_dia, param_turno, param_quirofano, param_cantidad]
            )
            
            btn_añadir.click(
                añadir_restriccion,
                inputs=[tipo_restr, param_esp, param_dia, param_turno, param_quirofano, param_cantidad],
                outputs=[lista_restricciones, msg_restriccion]
            )
            
            btn_eliminar.click(
                eliminar_restriccion,
                inputs=[input_eliminar],
                outputs=[lista_restricciones, msg_eliminar]
            )
            
            btn_limpiar.click(
                limpiar_restricciones,
                outputs=[lista_restricciones, msg_eliminar]
            )
        
        # =====================================================================
        # FASE 4: EJECUCIÓN OPERATIVA - ¿Qué hago hoy?
        # =====================================================================
        
        # TAB OPTIMIZACIÓN
        with gr.TabItem("⚙️ Optimizar"):
            gr.Markdown("""
            ### Optimización del Programa Quirúrgico
            ⚠️ **Respeta la configuración de sesiones** - Solo programa cirugías en quirófanos/turnos asignados a su especialidad.
            
            **Nuevas opciones v4.4:**
            - 🔮 **Reservas predictivas**: Reserva automática de tiempo para urgencias por especialidad (basado en ML)
            - 📊 **% Programación**: Controla qué porcentaje de la capacidad total programar
            """)
            
            with gr.Row():
                sl_peso = gr.Slider(20, 80, 60, label="Peso Clínico %", 
                                   info="Mayor = prioriza urgentes. Menor = prioriza eficiencia")
                dd_met = gr.Dropdown(
                    choices=['auto', 'heuristico', 'genetico', 'milp'],
                    value='auto', 
                    label="Método de Optimización",
                    info="auto=mejor disponible, genetico=algoritmo evolutivo, milp=óptimo matemático"
                )
            
            with gr.Row():
                chk_reservas = gr.Checkbox(
                    value=True, 
                    label="🔮 Usar Reservas Predictivas (ML)",
                    info="Reserva tiempo para urgencias según predicción por especialidad"
                )
                sl_pct_prog = gr.Slider(
                    50, 100, 85, 
                    label="% Capacidad a Programar",
                    info="Dejar margen para variabilidad y urgencias adicionales"
                )
            
            btn_opt = gr.Button("🚀 OPTIMIZAR", variant="primary", size="lg")
            opt_md = gr.Markdown()
            with gr.Row():
                opt_p1 = gr.Plot()
                opt_p2 = gr.Plot()
            btn_opt.click(optimizar, inputs=[sl_peso, dd_met, chk_reservas, sl_pct_prog], outputs=[opt_md, opt_p1, opt_p2])
        
        # TAB LISTA ESPERA (con filtros programadas/no programadas)
        with gr.TabItem("📋 Resultado"):
            filtro = gr.Radio(["Todos", "Solo programadas", "Solo NO programadas"], value="Todos")
            btn_lst = gr.Button("🔄 Ver", variant="primary")
            lst_md = gr.Markdown()
            lst_df = gr.Dataframe()
            lst_fig = gr.Plot()
            btn_lst.click(ver_lista_espera, inputs=[filtro], outputs=[lst_md, lst_df, lst_fig])
        
        # TAB PROGRAMA (ACTUALIZADO v4.6 - Vista Gantt)
        with gr.TabItem("📅 Programa"):
            gr.Markdown("""
            ### Vista del Programa Quirúrgico
            Visualización tipo Gantt del programa por quirófano. Muestra cirugías electivas, 
            tiempo libre y reservas para urgencias.
            """)
            
            with gr.Row():
                sl_dia_gantt = gr.Slider(0, 9, 0, step=1, label="Seleccionar Día", 
                                         info="Navega entre los días del horizonte")
            
            with gr.Row():
                chk_reservas = gr.Checkbox(value=True, label="🔶 Mostrar reservas urgencias")
                chk_urgencias = gr.Checkbox(value=True, label="🚨 Mostrar urgencias programadas")
            
            btn_gantt = gr.Button("📊 Ver Programa Gantt", variant="primary", size="lg")
            
            gantt_md = gr.Markdown()
            gantt_fig = gr.Plot(label="Programa por Quirófano")
            
            with gr.Accordion("📋 Detalle de Cirugías", open=False):
                gantt_tabla = gr.Dataframe(label="Lista de cirugías del día")
            
            btn_gantt.click(
                generar_vista_gantt,
                inputs=[sl_dia_gantt, chk_reservas, chk_urgencias],
                outputs=[gantt_md, gantt_fig, gantt_tabla]
            )
            
            gr.Markdown("""
            ---
            ### Leyenda de Colores
            | Color | Significado |
            |-------|-------------|
            | 🟦 Color sólido | Tiempo de cirugía |
            | 🟦 Color claro rayado | Tiempo de limpieza/preparación (30 min) |
            | 🟩 Verde claro | Tiempo libre (programable) |
            | 🟧 Naranja rayado | Reserva para urgencias (ML) |
            | 🟥 Rojo | Urgencia diferida programada |
            """)
        
        # TAB URGENCIAS DIFERIDAS
        with gr.TabItem("⚡ Urgencias"):
            gr.Markdown("""
            ### Gestión de Urgencias Diferidas
            
            Registra y gestiona las urgencias diferidas (no inmediatas) que deben operarse en 24-72h.
            El sistema las asignará automáticamente a los huecos reservados en el programa.
            """)
            
            with gr.Tabs():
                with gr.TabItem("📋 Lista Pendientes"):
                    with gr.Row():
                        btn_ver_urg = gr.Button("🔄 Actualizar Lista", variant="secondary")
                        btn_programar_urg = gr.Button("📅 Programar Urgencias", variant="primary", size="lg")
                    
                    urg_programacion_md = gr.Markdown()
                    
                    gr.Markdown("---")
                    urg_md = gr.Markdown()
                    urg_df = gr.Dataframe()
                    urg_fig = gr.Plot()
                    
                    btn_ver_urg.click(ver_urgencias_pendientes, outputs=[urg_md, urg_df, urg_fig])
                    
                    def programar_y_mostrar():
                        resumen, prog, sin = programar_urgencias_automaticamente()
                        md, df, fig = ver_urgencias_pendientes()
                        return resumen, md, df, fig
                    
                    btn_programar_urg.click(
                        programar_y_mostrar,
                        outputs=[urg_programacion_md, urg_md, urg_df, urg_fig]
                    )
                    
                    gr.Markdown("---\n### Acciones Manuales")
                    with gr.Row():
                        urg_id_input = gr.Textbox(label="ID Urgencia (ej: URG-0001)", placeholder="URG-XXXX")
                        btn_operar = gr.Button("✅ Marcar Operada", variant="secondary")
                        btn_cancelar = gr.Button("❌ Cancelar", variant="stop")
                    
                    urg_accion_msg = gr.Markdown()
                    
                    btn_operar.click(marcar_urgencia_operada, inputs=[urg_id_input], 
                                    outputs=[urg_accion_msg, urg_md, urg_df, urg_fig])
                    btn_cancelar.click(cancelar_urgencia, inputs=[urg_id_input], 
                                      outputs=[urg_accion_msg, urg_md, urg_df, urg_fig])
                
                with gr.TabItem("➕ Registrar Nueva"):
                    gr.Markdown("**Registrar una nueva urgencia diferida**")
                    
                    with gr.Row():
                        with gr.Column():
                            nueva_paciente = gr.Textbox(label="Nombre Paciente *", placeholder="Juan García López")
                            nueva_edad = gr.Number(label="Edad", value=50, minimum=1, maximum=100)
                            nueva_esp = gr.Dropdown(
                                choices=[(ESPECIALIDADES_NOMBRES.get(e, e), e) for e in ESPECIALIDADES_SIN_LIBRE],
                                label="Especialidad *",
                                value="CIRUGIA_GENERAL"
                            )
                        with gr.Column():
                            nueva_diag = gr.Textbox(label="Diagnóstico *", placeholder="Apendicitis aguda")
                            nueva_proc = gr.Textbox(label="Procedimiento *", placeholder="Apendicectomía laparoscópica")
                            nueva_dur = gr.Number(label="Duración estimada (min)", value=60, minimum=15, maximum=480)
                    
                    with gr.Row():
                        nueva_horas = gr.Radio(
                            choices=[(24, 24), (48, 48), (72, 72)],
                            value=24,
                            label="Tiempo Límite (horas)",
                            info="Máximo tiempo para operar"
                        )
                        nueva_prio = gr.Radio(
                            choices=[(1, 1), (2, 2), (3, 3)],
                            value=2,
                            label="Prioridad",
                            info="1=Máxima, 2=Alta, 3=Media"
                        )
                    
                    nueva_notas = gr.Textbox(label="Notas (opcional)", placeholder="Información adicional...")
                    
                    btn_registrar = gr.Button("💾 Registrar Urgencia", variant="primary", size="lg")
                    registro_msg = gr.Markdown()
                    registro_md = gr.Markdown()
                    registro_df = gr.Dataframe()
                    registro_fig = gr.Plot()
                    
                    btn_registrar.click(
                        agregar_urgencia_ui,
                        inputs=[nueva_paciente, nueva_edad, nueva_esp, nueva_diag, 
                               nueva_proc, nueva_dur, nueva_horas, nueva_prio, nueva_notas],
                        outputs=[registro_msg, registro_md, registro_df, registro_fig]
                    )
                
                with gr.TabItem("🔄 Demo"):
                    gr.Markdown("**Regenerar urgencias de ejemplo** para demostración")
                    btn_reset_urg = gr.Button("🔄 Regenerar Ejemplos", variant="secondary")
                    reset_msg = gr.Markdown()
                    reset_md = gr.Markdown()
                    reset_df = gr.Dataframe()
                    reset_fig = gr.Plot()
                    
                    btn_reset_urg.click(reset_urgencias_ejemplo, 
                                        outputs=[reset_msg, reset_md, reset_df, reset_fig])
        
        # TAB VISTA CALENDARIO
        with gr.TabItem("📆 Calendario"):
            gr.Markdown("""
            ### Vista Calendario del Programa
            
            Visualización del programa quirúrgico organizado por semanas.
            **Ejecuta primero una optimización** en la pestaña "⚙️ Optimizar".
            """)
            
            btn_calendario = gr.Button("📅 Generar Vista Calendario", variant="primary", size="lg")
            cal_md = gr.Markdown()
            
            with gr.Row():
                cal_heatmap = gr.Plot(label="Ocupación por Quirófano/Día")
                cal_barras = gr.Plot(label="Distribución por Especialidad")
            
            btn_calendario.click(generar_vista_calendario, outputs=[cal_md, cal_heatmap, cal_barras])
        
        # =====================================================================
        # FASE 5: TÉCNICO
        # =====================================================================
        
        # TAB ML
        with gr.TabItem("🤖 ML"):
            btn_ml = gr.Button("Ver ML", variant="primary")
            ml_md = gr.Markdown()
            ml_df = gr.Dataframe()
            ml_fig = gr.Plot()
            btn_ml.click(ver_ml, outputs=[ml_md, ml_df, ml_fig])
    
    gr.Markdown("---\n**v4.9** • Flujo: Análisis → Planificación → Config → Ejecución • **What-If + Monte Carlo** • CatSalut")

print("="*50 + "\n🚀 LANZANDO...\n" + "="*50)
demo.launch(share=True)
