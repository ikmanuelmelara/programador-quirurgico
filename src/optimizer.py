"""
Motor de OptimizaciÃ³n del Programador QuirÃºrgico
================================================
Implementa algoritmos de optimizaciÃ³n para la programaciÃ³n quirÃºrgica.
"""

import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import random
import time

from config import (
    PesosOptimizacion, PESOS_DEFAULT, RestriccionesGlobales, 
    RESTRICCIONES_DEFAULT, ConfiguracionSistema, CONFIG_DEFAULT,
    QUIROFANOS_DEFAULT, Quirofano, PrioridadCatSalut
)
from models import (
    SolicitudCirugia, CirugiaProgramada, ProgramaDiario, ProgramaPeriodo,
    Cirujano, generar_id_cirugia
)
from constraint_learning import RestriccionAprendida


@dataclass
class ResultadoOptimizacion:
    """Resultado de una ejecuciÃ³n del optimizador"""
    programa: ProgramaPeriodo
    score_total: float
    score_clinico: float
    score_eficiencia: float
    cirugias_programadas: int
    cirugias_no_programadas: int
    restricciones_violadas: List[str]
    tiempo_ejecucion_seg: float
    iteraciones: int
    convergencia: List[float]


class SolucionParcial:
    """Representa una soluciÃ³n parcial durante la optimizaciÃ³n"""
    
    def __init__(self, periodo: ProgramaPeriodo):
        self.periodo = periodo
        self.asignaciones: List[CirugiaProgramada] = []
    
    def agregar(self, cirugia: CirugiaProgramada) -> bool:
        programa_dia = self.periodo.obtener_dia(cirugia.fecha)
        if programa_dia.agregar_cirugia(cirugia):
            self.asignaciones.append(cirugia)
            return True
        return False
    
    def clonar(self) -> 'SolucionParcial':
        nueva = SolucionParcial(
            ProgramaPeriodo(
                fecha_inicio=self.periodo.fecha_inicio,
                fecha_fin=self.periodo.fecha_fin
            )
        )
        for c in self.asignaciones:
            nueva_c = CirugiaProgramada(
                id=c.id, solicitud=c.solicitud, fecha=c.fecha,
                hora_inicio=c.hora_inicio, duracion_programada_min=c.duracion_programada_min,
                quirofano_id=c.quirofano_id, cirujano=c.cirujano
            )
            nueva.agregar(nueva_c)
        return nueva


class OptimizadorQuirurgico:
    """Motor principal de optimizaciÃ³n para programaciÃ³n quirÃºrgica."""
    
    def __init__(
        self,
        pesos: PesosOptimizacion = None,
        restricciones: RestriccionesGlobales = None,
        config: ConfiguracionSistema = None,
        quirofanos: List[Quirofano] = None,
        restricciones_aprendidas: List[RestriccionAprendida] = None
    ):
        self.pesos = pesos or PESOS_DEFAULT
        self.restricciones = restricciones or RESTRICCIONES_DEFAULT
        self.config = config or CONFIG_DEFAULT
        self.quirofanos = quirofanos or QUIROFANOS_DEFAULT
        self.restricciones_aprendidas = restricciones_aprendidas or []
        self._convergencia: List[float] = []
        
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
    
    def optimizar(
        self,
        solicitudes: List[SolicitudCirugia],
        cirujanos: List[Cirujano],
        fecha_inicio: date = None,
        fecha_fin: date = None,
        metodo: str = 'hibrido'
    ) -> ResultadoOptimizacion:
        """Ejecuta la optimizaciÃ³n completa."""
        inicio = time.time()
        self._convergencia = []
        
        if fecha_inicio is None:
            fecha_inicio = date.today() + timedelta(days=1)
        if fecha_fin is None:
            fecha_fin = fecha_inicio + timedelta(days=self.config.horizonte_dias)
        
        print("=" * 60)
        print("OPTIMIZACIÃ“N DE PROGRAMA QUIRÃšRGICO")
        print("=" * 60)
        print(f"Horizonte: {fecha_inicio} a {fecha_fin}")
        print(f"Solicitudes: {len(solicitudes)}")
        print(f"Balance: ClÃ­nico={self.pesos.peso_prioridad_clinica:.0%}, Eficiencia={self.pesos.peso_eficiencia_operativa:.0%}")
        
        # Filtrar vÃ¡lidas
        solicitudes_validas = [
            s for s in solicitudes 
            if s.activa and not s.cancelada and s.preoperatorio_completado
        ]
        print(f"Solicitudes vÃ¡lidas: {len(solicitudes_validas)}")
        
        # Ordenar por prioridad
        solicitudes_ordenadas = self._ordenar_por_prioridad(solicitudes_validas)
        
        # Optimizar
        solucion = self._optimizar_hibrido(
            solicitudes_ordenadas, cirujanos, fecha_inicio, fecha_fin
        )
        
        tiempo_total = time.time() - inicio
        
        # MÃ©tricas finales
        score_total, score_clinico, score_eficiencia = self._calcular_scores(
            solucion, solicitudes_ordenadas
        )
        
        violaciones = self._verificar_restricciones(solucion)
        ids_programadas = {c.solicitud.id for c in solucion.asignaciones}
        
        resultado = ResultadoOptimizacion(
            programa=solucion.periodo,
            score_total=score_total,
            score_clinico=score_clinico,
            score_eficiencia=score_eficiencia,
            cirugias_programadas=len(solucion.asignaciones),
            cirugias_no_programadas=len(solicitudes_validas) - len(ids_programadas),
            restricciones_violadas=violaciones,
            tiempo_ejecucion_seg=tiempo_total,
            iteraciones=len(self._convergencia),
            convergencia=self._convergencia.copy()
        )
        
        print(f"\n=> Completado en {tiempo_total:.1f}s")
        print(f"   Score: {score_total:.4f} | Programadas: {resultado.cirugias_programadas}")
        
        return resultado
    
    def _ordenar_por_prioridad(self, solicitudes: List[SolicitudCirugia]) -> List[SolicitudCirugia]:
        """Ordena por criterios CatSalut."""
        def score(s):
            pesos = {
                PrioridadCatSalut.URGENTE: 1000,
                PrioridadCatSalut.ONCOLOGICO_PRIORITARIO: 100,
                PrioridadCatSalut.ONCOLOGICO_ESTANDAR: 90,
                PrioridadCatSalut.CARDIACA: 80,
                PrioridadCatSalut.REFERENCIA_P1: 60,
                PrioridadCatSalut.REFERENCIA_P2: 40,
                PrioridadCatSalut.REFERENCIA_P3: 20,
            }
            base = pesos.get(s.prioridad, 30)
            tiempo_bonus = s.porcentaje_tiempo_consumido * 50
            fuera_plazo = 100 if s.esta_fuera_plazo else 0
            return base + tiempo_bonus + fuera_plazo + s.score_clinico * 0.5
        
        return sorted(solicitudes, key=score, reverse=True)
    
    def _optimizar_hibrido(
        self,
        solicitudes: List[SolicitudCirugia],
        cirujanos: List[Cirujano],
        fecha_inicio: date,
        fecha_fin: date
    ) -> SolucionParcial:
        """Combina heurÃ­stica + bÃºsqueda local."""
        # Fase 1: HeurÃ­stica constructiva
        print("\n--- Fase HeurÃ­stica ---")
        solucion = self._heuristica_constructiva(
            solicitudes, cirujanos, fecha_inicio, fecha_fin
        )
        
        # Fase 2: BÃºsqueda local
        print("\n--- Fase BÃºsqueda Local ---")
        solucion = self._busqueda_local(solucion, solicitudes, cirujanos)
        
        return solucion
    
    def _heuristica_constructiva(
        self,
        solicitudes: List[SolicitudCirugia],
        cirujanos: List[Cirujano],
        fecha_inicio: date,
        fecha_fin: date
    ) -> SolucionParcial:
        """First Fit Decreasing adaptado."""
        periodo = ProgramaPeriodo(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
        solucion = SolucionParcial(periodo)
        
        # DÃ­as hÃ¡biles
        dias = []
        dia = fecha_inicio
        while dia <= fecha_fin:
            if dia.weekday() in self.config.dias_operativos:
                dias.append(dia)
            dia += timedelta(days=1)
        
        # Estado quirÃ³fanos
        estado = {}
        for d in dias:
            for q in self.quirofanos:
                if q.activo:
                    estado[(d, q.id)] = q.horario_inicio
        
        programadas = 0
        for idx, sol in enumerate(solicitudes):
            # Asignar cirujano si falta
            if not sol.cirujano_asignado:
                cirujano = self._encontrar_cirujano(sol, cirujanos)
                if not cirujano:
                    continue
                sol.cirujano_asignado = cirujano
            
            mejor_slot = None
            mejor_score = -float('inf')
            
            for dia in dias:
                if dia.weekday() not in sol.cirujano_asignado.dias_disponibles:
                    continue
                
                for q in self.quirofanos:
                    if not q.activo:
                        continue
                    if sol.tipo_intervencion.especialidad not in q.especialidades_permitidas:
                        continue
                    
                    # Verificar equipamiento
                    if not set(sol.tipo_intervencion.requiere_equipo_especial).issubset(set(q.equipamiento_especial)):
                        continue
                    
                    hora = estado.get((dia, q.id), q.horario_inicio)
                    duracion = sol.duracion_esperada()
                    
                    if hora + duracion > q.horario_fin + self.restricciones.max_overtime_permitido_min:
                        continue
                    
                    # Evaluar slot
                    score = self._evaluar_slot(sol, dia, hora, q.id)
                    if score > mejor_score:
                        mejor_score = score
                        mejor_slot = (dia, hora, q.id, duracion)
            
            if mejor_slot:
                dia, hora, qid, dur = mejor_slot
                cirugia = CirugiaProgramada(
                    id=generar_id_cirugia(),
                    solicitud=sol,
                    fecha=dia,
                    hora_inicio=hora,
                    duracion_programada_min=dur,
                    quirofano_id=qid,
                    cirujano=sol.cirujano_asignado
                )
                if solucion.agregar(cirugia):
                    estado[(dia, qid)] = hora + dur + self.restricciones.tiempo_limpieza_entre_cirugias_min
                    programadas += 1
            
            if (idx + 1) % 50 == 0:
                print(f"   Procesadas {idx+1}/{len(solicitudes)}, programadas: {programadas}")
        
        print(f"   Total: {programadas}/{len(solicitudes)}")
        self._convergencia.append(self._calcular_score_total(solucion, solicitudes))
        return solucion
    
    def _busqueda_local(
        self,
        solucion: SolucionParcial,
        solicitudes: List[SolicitudCirugia],
        cirujanos: List[Cirujano],
        max_iter: int = 100
    ) -> SolucionParcial:
        """Mejora iterativa."""
        mejor_score = self._calcular_score_total(solucion, solicitudes)
        mejor = solucion
        sin_mejora = 0
        
        for it in range(max_iter):
            vecino = self._generar_vecino(mejor)
            score = self._calcular_score_total(vecino, solicitudes)
            
            if score > mejor_score:
                mejor_score = score
                mejor = vecino
                sin_mejora = 0
            else:
                sin_mejora += 1
            
            self._convergencia.append(mejor_score)
            
            if sin_mejora > 20:
                break
        
        print(f"   Iteraciones: {it+1}, Score: {mejor_score:.4f}")
        return mejor
    
    def _generar_vecino(self, solucion: SolucionParcial) -> SolucionParcial:
        """Genera soluciÃ³n vecina."""
        vecino = solucion.clonar()
        if not vecino.asignaciones:
            return vecino
        
        # Mover una cirugÃ­a aleatoriamente
        cirugia = random.choice(vecino.asignaciones)
        delta = random.randint(-30, 30)
        cirugia.hora_inicio = max(8*60, min(cirugia.hora_inicio + delta, 14*60))
        
        return vecino
    
    def _encontrar_cirujano(self, sol: SolicitudCirugia, cirujanos: List[Cirujano]) -> Optional[Cirujano]:
        """Encuentra cirujano apropiado."""
        validos = [c for c in cirujanos if sol.tipo_intervencion.especialidad == c.especialidad_principal]
        return random.choice(validos) if validos else None
    
    def _evaluar_slot(self, sol: SolicitudCirugia, fecha: date, hora: int, qid: int) -> float:
        """EvalÃºa calidad de un slot."""
        score = 100.0
        
        # OncolÃ³gico temprano
        if sol.prioridad in [PrioridadCatSalut.ONCOLOGICO_PRIORITARIO, PrioridadCatSalut.ONCOLOGICO_ESTANDAR]:
            if hora > 10 * 60:
                score -= 20
        
        # Bonus por programar pronto si fuera de plazo
        if sol.esta_fuera_plazo:
            score += 30
        
        # QuirÃ³fano preferido
        if sol.cirujano_asignado and qid in sol.cirujano_asignado.quirofanos_preferidos:
            score += 10
        
        # Restricciones aprendidas
        for r in self.restricciones_aprendidas:
            if r.tipo == 'preferencia_quirofano':
                if sol.cirujano_asignado and r.entidades.get('cirujano_id') == sol.cirujano_asignado.id:
                    if r.entidades.get('quirofano_id') == qid:
                        score += 15 * r.confianza
        
        return score
    
    def _calcular_score_total(self, solucion: SolucionParcial, solicitudes: List[SolicitudCirugia]) -> float:
        """Score total."""
        sc = self._score_clinico(solucion, solicitudes)
        se = self._score_eficiencia(solucion)
        return self.pesos.peso_prioridad_clinica * sc + self.pesos.peso_eficiencia_operativa * se
    
    def _calcular_scores(self, solucion: SolucionParcial, solicitudes: List[SolicitudCirugia]) -> Tuple[float, float, float]:
        sc = self._score_clinico(solucion, solicitudes)
        se = self._score_eficiencia(solucion)
        return (self.pesos.peso_prioridad_clinica * sc + self.pesos.peso_eficiencia_operativa * se, sc, se)
    
    def _score_clinico(self, solucion: SolucionParcial, solicitudes: List[SolicitudCirugia]) -> float:
        if not solicitudes:
            return 0.0
        
        score = 0.0
        ids_prog = {c.solicitud.id for c in solucion.asignaciones}
        
        for s in solicitudes:
            if s.id in ids_prog:
                if s.prioridad == PrioridadCatSalut.ONCOLOGICO_PRIORITARIO:
                    score += 10
                elif s.prioridad == PrioridadCatSalut.ONCOLOGICO_ESTANDAR:
                    score += 8
                elif s.esta_fuera_plazo:
                    score += 7
                else:
                    score += 4
            else:
                if s.prioridad in [PrioridadCatSalut.ONCOLOGICO_PRIORITARIO, PrioridadCatSalut.ONCOLOGICO_ESTANDAR]:
                    score -= 5
        
        return max(0, score / (len(solicitudes) * 10))
    
    def _score_eficiencia(self, solucion: SolucionParcial) -> float:
        if not solucion.asignaciones:
            return 0.0
        
        scores = []
        for programa in solucion.periodo.programas_diarios.values():
            for q in self.quirofanos:
                if q.activo:
                    util = programa.utilizacion_quirofano(q.id, q.horario_inicio, q.horario_fin)
                    if util > 0:
                        scores.append(util)
        
        return np.mean(scores) if scores else 0
    
    def _verificar_restricciones(self, solucion: SolucionParcial) -> List[str]:
        """Verifica violaciones."""
        violaciones = []
        
        for fecha, programa in solucion.periodo.programas_diarios.items():
            uci = programa.ingresos_uci_esperados()
            if uci > self.restricciones.max_ingresos_uci_dia:
                violaciones.append(f"{fecha}: Exceso UCI ({uci:.1f})")
            
            for q in self.quirofanos:
                ot = programa.overtime_quirofano(q.id, q.horario_fin)
                if ot > self.restricciones.max_overtime_permitido_min:
                    violaciones.append(f"{fecha} Q{q.id}: Overtime ({ot}min)")
        
        return violaciones


def main():
    """Prueba del optimizador"""
    from synthetic_data import GeneradorDatosSinteticos
    from constraint_learning import AprendizajeRestricciones
    
    print("Generando datos...")
    gen = GeneradorDatosSinteticos(seed=42)
    cirujanos, lista_espera, historico = gen.generar_dataset_completo(
        n_solicitudes_espera=150, dias_historico=180
    )
    
    print("\nAprendiendo restricciones...")
    aprendizaje = AprendizajeRestricciones()
    restricciones = aprendizaje.analizar_historico(historico)
    
    print("\nOptimizando...")
    opt = OptimizadorQuirurgico(restricciones_aprendidas=restricciones)
    resultado = opt.optimizar(lista_espera, cirujanos)
    
    print("\n" + "=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Score: {resultado.score_total:.4f}")
    print(f"Programadas: {resultado.cirugias_programadas}")
    print(f"No programadas: {resultado.cirugias_no_programadas}")
    
    return resultado


if __name__ == "__main__":
    main()
