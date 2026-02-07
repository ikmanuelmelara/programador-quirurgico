"""
Programador QuirÃºrgico Inteligente - AplicaciÃ³n Principal
==========================================================
Sistema completo de optimizaciÃ³n de programaciÃ³n quirÃºrgica
para un bloque de cirugÃ­a general con 8 quirÃ³fanos.

CaracterÃ­sticas:
- Criterios de priorizaciÃ³n CatSalut (Catalunya)
- Aprendizaje automÃ¡tico de restricciones
- Balance configurable prioridad clÃ­nica / eficiencia operativa
- Datos sintÃ©ticos realistas para demostraciÃ³n
"""

import json
from datetime import date, timedelta
from typing import Dict, List, Any
import pandas as pd

from config import (
    PesosOptimizacion, PESOS_DEFAULT, PrioridadCatSalut,
    TIEMPOS_MAXIMOS_ESPERA, Especialidad, QUIROFANOS_DEFAULT
)
from models import SolicitudCirugia, Cirujano, ProgramaPeriodo
from synthetic_data import GeneradorDatosSinteticos
from constraint_learning import AprendizajeRestricciones
from optimizer import OptimizadorQuirurgico, ResultadoOptimizacion


class ProgramadorQuirurgico:
    """
    Clase principal que integra todos los componentes del sistema.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generador = GeneradorDatosSinteticos(seed=seed)
        self.aprendizaje = AprendizajeRestricciones()
        
        # Datos
        self.cirujanos: List[Cirujano] = []
        self.lista_espera: List[SolicitudCirugia] = []
        self.historico: pd.DataFrame = None
        self.restricciones_aprendidas = []
        
        # ConfiguraciÃ³n de pesos (configurable por usuario)
        self.pesos = PESOS_DEFAULT
        
        # Resultado
        self.ultimo_resultado: ResultadoOptimizacion = None
    
    def inicializar_datos_sinteticos(
        self,
        n_solicitudes: int = 250,
        dias_historico: int = 365
    ):
        """Genera todos los datos sintÃ©ticos necesarios."""
        print("=" * 70)
        print("INICIALIZANDO PROGRAMADOR QUIRÃšRGICO INTELIGENTE")
        print("=" * 70)
        
        self.cirujanos, self.lista_espera, self.historico = \
            self.generador.generar_dataset_completo(
                n_solicitudes_espera=n_solicitudes,
                dias_historico=dias_historico
            )
        
        # Aprender restricciones del histÃ³rico
        self.restricciones_aprendidas = self.aprendizaje.analizar_historico(
            self.historico
        )
        
        print("\n" + self.aprendizaje.generar_resumen())
    
    def configurar_pesos(
        self,
        peso_clinico: float = 0.6,
        peso_eficiencia: float = 0.4
    ):
        """
        Configura el balance entre prioridad clÃ­nica y eficiencia operativa.
        
        Args:
            peso_clinico: Peso para prioridad clÃ­nica (0-1)
            peso_eficiencia: Peso para eficiencia operativa (0-1)
        """
        if abs(peso_clinico + peso_eficiencia - 1.0) > 0.001:
            raise ValueError("Los pesos deben sumar 1.0")
        
        self.pesos = PesosOptimizacion(
            peso_prioridad_clinica=peso_clinico,
            peso_eficiencia_operativa=peso_eficiencia
        )
        
        print(f"Pesos configurados: ClÃ­nico={peso_clinico:.0%}, Eficiencia={peso_eficiencia:.0%}")
    
    def optimizar_programa(
        self,
        fecha_inicio: date = None,
        horizonte_dias: int = 14
    ) -> ResultadoOptimizacion:
        """
        Ejecuta la optimizaciÃ³n del programa quirÃºrgico.
        
        Args:
            fecha_inicio: Fecha de inicio (None = maÃ±ana)
            horizonte_dias: DÃ­as a programar
        
        Returns:
            ResultadoOptimizacion con el programa optimizado
        """
        if not self.lista_espera:
            raise ValueError("No hay datos. Ejecute inicializar_datos_sinteticos() primero.")
        
        if fecha_inicio is None:
            fecha_inicio = date.today() + timedelta(days=1)
        
        fecha_fin = fecha_inicio + timedelta(days=horizonte_dias)
        
        optimizador = OptimizadorQuirurgico(
            pesos=self.pesos,
            restricciones_aprendidas=self.restricciones_aprendidas
        )
        
        self.ultimo_resultado = optimizador.optimizar(
            solicitudes=self.lista_espera,
            cirujanos=self.cirujanos,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin
        )
        
        return self.ultimo_resultado
    
    def generar_informe(self) -> str:
        """Genera un informe textual del resultado."""
        if not self.ultimo_resultado:
            return "No hay resultado. Ejecute optimizar_programa() primero."
        
        r = self.ultimo_resultado
        
        lineas = [
            "=" * 70,
            "INFORME DE PROGRAMA QUIRÃšRGICO OPTIMIZADO",
            "=" * 70,
            "",
            "CONFIGURACIÃ“N:",
            f"  â€¢ Peso prioridad clÃ­nica: {self.pesos.peso_prioridad_clinica:.0%}",
            f"  â€¢ Peso eficiencia operativa: {self.pesos.peso_eficiencia_operativa:.0%}",
            "",
            "MÃ‰TRICAS DE OPTIMIZACIÃ“N:",
            f"  â€¢ Score total: {r.score_total:.4f}",
            f"  â€¢ Score clÃ­nico: {r.score_clinico:.4f}",
            f"  â€¢ Score eficiencia: {r.score_eficiencia:.4f}",
            "",
            "RESUMEN DEL PROGRAMA:",
            f"  â€¢ CirugÃ­as programadas: {r.cirugias_programadas}",
            f"  â€¢ CirugÃ­as pendientes: {r.cirugias_no_programadas}",
            f"  â€¢ Tiempo de optimizaciÃ³n: {r.tiempo_ejecucion_seg:.1f} segundos",
            f"  â€¢ Iteraciones: {r.iteraciones}",
            "",
        ]
        
        if r.restricciones_violadas:
            lineas.append("ALERTAS:")
            for v in r.restricciones_violadas[:10]:
                lineas.append(f"  âš  {v}")
            if len(r.restricciones_violadas) > 10:
                lineas.append(f"  ... y {len(r.restricciones_violadas) - 10} mÃ¡s")
        
        # Resumen por dÃ­a
        lineas.extend(["", "PROGRAMA POR DÃA:"])
        dias_semana = ['Lun', 'Mar', 'MiÃ©', 'Jue', 'Vie', 'SÃ¡b', 'Dom']
        
        for fecha in sorted(r.programa.programas_diarios.keys()):
            prog = r.programa.programas_diarios[fecha]
            dia_sem = dias_semana[fecha.weekday()]
            resumen = prog.resumen()
            
            lineas.append(
                f"  {fecha} ({dia_sem}): {resumen['total_cirugias']} cirugÃ­as, "
                f"{resumen['cirugias_oncologicas']} oncolÃ³gicas, "
                f"util. {resumen['utilizacion_media']*100:.0f}%"
            )
        
        return "\n".join(lineas)
    
    def exportar_programa_json(self) -> Dict[str, Any]:
        """Exporta el programa en formato JSON."""
        if not self.ultimo_resultado:
            return {"error": "No hay resultado"}
        
        programa = []
        for fecha, prog in self.ultimo_resultado.programa.programas_diarios.items():
            for c in prog.cirugias:
                programa.append({
                    'fecha': fecha.isoformat(),
                    'hora_inicio': c.hora_inicio_str,
                    'hora_fin': c.hora_fin_str,
                    'quirofano': c.quirofano_id,
                    'paciente': c.solicitud.paciente.nombre,
                    'intervencion': c.solicitud.tipo_intervencion.nombre,
                    'cirujano': c.cirujano.nombre if c.cirujano else 'Sin asignar',
                    'prioridad': c.solicitud.prioridad.name,
                    'especialidad': c.solicitud.tipo_intervencion.especialidad.name,
                })
        
        return {
            'configuracion': {
                'peso_clinico': self.pesos.peso_prioridad_clinica,
                'peso_eficiencia': self.pesos.peso_eficiencia_operativa,
            },
            'metricas': {
                'score_total': self.ultimo_resultado.score_total,
                'score_clinico': self.ultimo_resultado.score_clinico,
                'score_eficiencia': self.ultimo_resultado.score_eficiencia,
                'cirugias_programadas': self.ultimo_resultado.cirugias_programadas,
                'tiempo_optimizacion': self.ultimo_resultado.tiempo_ejecucion_seg,
            },
            'programa': programa,
        }
    
    def estadisticas_lista_espera(self) -> Dict[str, Any]:
        """Genera estadÃ­sticas de la lista de espera."""
        if not self.lista_espera:
            return {}
        
        total = len(self.lista_espera)
        fuera_plazo = sum(1 for s in self.lista_espera if s.esta_fuera_plazo)
        
        por_prioridad = {}
        for p in PrioridadCatSalut:
            count = sum(1 for s in self.lista_espera if s.prioridad == p)
            if count > 0:
                por_prioridad[p.name] = {
                    'cantidad': count,
                    'porcentaje': count / total * 100,
                    'tiempo_max_dias': TIEMPOS_MAXIMOS_ESPERA.get(p, 365)
                }
        
        por_especialidad = {}
        for e in Especialidad:
            count = sum(1 for s in self.lista_espera 
                       if s.tipo_intervencion.especialidad == e)
            if count > 0:
                por_especialidad[e.name] = {
                    'cantidad': count,
                    'porcentaje': count / total * 100
                }
        
        dias_espera = [s.dias_en_espera for s in self.lista_espera]
        
        return {
            'total_solicitudes': total,
            'fuera_de_plazo': fuera_plazo,
            'porcentaje_fuera_plazo': fuera_plazo / total * 100,
            'dias_espera_promedio': sum(dias_espera) / len(dias_espera),
            'dias_espera_maximo': max(dias_espera),
            'por_prioridad': por_prioridad,
            'por_especialidad': por_especialidad,
        }


def demo_interactiva():
    """DemostraciÃ³n interactiva del sistema."""
    print("\n" + "=" * 70)
    print("  DEMO: PROGRAMADOR QUIRÃšRGICO INTELIGENTE")
    print("  Sistema de OptimizaciÃ³n para Bloque QuirÃºrgico - 8 QuirÃ³fanos")
    print("  Basado en criterios de priorizaciÃ³n CatSalut (Catalunya)")
    print("=" * 70)
    
    # Inicializar
    programador = ProgramadorQuirurgico(seed=42)
    programador.inicializar_datos_sinteticos(n_solicitudes=200, dias_historico=365)
    
    # Mostrar estadÃ­sticas
    print("\n" + "=" * 70)
    print("ESTADÃSTICAS DE LISTA DE ESPERA")
    print("=" * 70)
    stats = programador.estadisticas_lista_espera()
    print(f"Total solicitudes: {stats['total_solicitudes']}")
    print(f"Fuera de plazo: {stats['fuera_de_plazo']} ({stats['porcentaje_fuera_plazo']:.1f}%)")
    print(f"DÃ­as espera promedio: {stats['dias_espera_promedio']:.0f}")
    print(f"DÃ­as espera mÃ¡ximo: {stats['dias_espera_maximo']}")
    
    print("\nPor prioridad:")
    for nombre, datos in stats['por_prioridad'].items():
        print(f"  {nombre}: {datos['cantidad']} ({datos['porcentaje']:.1f}%) - MÃ¡x: {datos['tiempo_max_dias']} dÃ­as")
    
    # Optimizar con balance por defecto (60% clÃ­nico, 40% eficiencia)
    print("\n" + "=" * 70)
    print("OPTIMIZACIÃ“N CON BALANCE ESTÃNDAR (60% ClÃ­nico / 40% Eficiencia)")
    print("=" * 70)
    resultado1 = programador.optimizar_programa(horizonte_dias=10)
    print(programador.generar_informe())
    
    # Optimizar con prioridad clÃ­nica alta
    print("\n" + "=" * 70)
    print("OPTIMIZACIÃ“N CON PRIORIDAD CLÃNICA ALTA (80% ClÃ­nico / 20% Eficiencia)")
    print("=" * 70)
    programador.configurar_pesos(peso_clinico=0.8, peso_eficiencia=0.2)
    resultado2 = programador.optimizar_programa(horizonte_dias=10)
    print(programador.generar_informe())
    
    # Optimizar con eficiencia alta
    print("\n" + "=" * 70)
    print("OPTIMIZACIÃ“N CON EFICIENCIA ALTA (40% ClÃ­nico / 60% Eficiencia)")
    print("=" * 70)
    programador.configurar_pesos(peso_clinico=0.4, peso_eficiencia=0.6)
    resultado3 = programador.optimizar_programa(horizonte_dias=10)
    print(programador.generar_informe())
    
    # ComparaciÃ³n
    print("\n" + "=" * 70)
    print("COMPARACIÃ“N DE CONFIGURACIONES")
    print("=" * 70)
    print(f"{'ConfiguraciÃ³n':<30} {'Score Total':>12} {'Score ClÃ­n.':>12} {'Score Efic.':>12} {'Programadas':>12}")
    print("-" * 78)
    print(f"{'60% ClÃ­nico / 40% Eficiencia':<30} {resultado1.score_total:>12.4f} {resultado1.score_clinico:>12.4f} {resultado1.score_eficiencia:>12.4f} {resultado1.cirugias_programadas:>12}")
    print(f"{'80% ClÃ­nico / 20% Eficiencia':<30} {resultado2.score_total:>12.4f} {resultado2.score_clinico:>12.4f} {resultado2.score_eficiencia:>12.4f} {resultado2.cirugias_programadas:>12}")
    print(f"{'40% ClÃ­nico / 60% Eficiencia':<30} {resultado3.score_total:>12.4f} {resultado3.score_clinico:>12.4f} {resultado3.score_eficiencia:>12.4f} {resultado3.cirugias_programadas:>12}")
    
    return programador


if __name__ == "__main__":
    programador = demo_interactiva()
