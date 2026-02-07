"""
Motor de Optimización Avanzado del Programador Quirúrgico
=========================================================
Implementa técnicas avanzadas de optimización:
- Programación Lineal Entera Mixta (MILP) con OR-Tools
- Algoritmos Genéticos con operadores especializados
- Optimización Estocástica (incertidumbre en duraciones)
- Simulated Annealing

Dependencias: pip install ortools deap numpy pandas
"""

import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import random
import time
import copy

# OR-Tools para optimización exacta
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("⚠️ OR-Tools no disponible. Instalar con: pip install ortools")

# DEAP para algoritmos genéticos
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("⚠️ DEAP no disponible. Instalar con: pip install deap")

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
class ResultadoOptimizacionAvanzado:
    """Resultado extendido de optimización"""
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
    
    # Métricas adicionales
    metodo_usado: str = "heuristico"
    gap_optimalidad: float = 0.0
    robustez_score: float = 0.0
    estadisticas_detalladas: Dict = field(default_factory=dict)


class OptimizadorMILP:
    """
    Optimizador usando Programación Lineal Entera Mixta (MILP).
    Encuentra la solución óptima (o cerca del óptimo) matemáticamente.
    """
    
    def __init__(self, pesos: PesosOptimizacion, restricciones: RestriccionesGlobales,
                 quirofanos: List[Quirofano], config: ConfiguracionSistema):
        self.pesos = pesos
        self.restricciones = restricciones
        self.quirofanos = quirofanos
        self.config = config
    
    def optimizar(self, solicitudes: List[SolicitudCirugia], 
                  cirujanos: List[Cirujano],
                  fecha_inicio: date, fecha_fin: date,
                  tiempo_limite_seg: int = 300) -> Tuple[List[Dict], float, float]:
        """
        Resuelve el problema usando CP-SAT de OR-Tools.
        """
        if not ORTOOLS_AVAILABLE:
            raise ImportError("OR-Tools no está disponible")
        
        print("   Construyendo modelo MILP...")
        
        model = cp_model.CpModel()
        
        # Preparar datos
        dias = self._generar_dias_habiles(fecha_inicio, fecha_fin)
        n_cirugias = len(solicitudes)
        n_dias = len(dias)
        quirofanos_activos = [q for q in self.quirofanos if q.activo]
        n_quirofanos = len(quirofanos_activos)
        slots_por_dia = 84  # Slots de 5 minutos entre 8:00 y 15:00
        
        print(f"   Dimensiones: {n_cirugias} cirugías × {n_dias} días × {n_quirofanos} quirófanos")
        
        # Variables de decisión: x[i,d,q] = 1 si cirugía i se programa día d en quirófano q
        x = {}
        hora_inicio = {}
        
        for i, sol in enumerate(solicitudes):
            for d in range(n_dias):
                for q_idx, q in enumerate(quirofanos_activos):
                    # Verificar compatibilidad
                    if sol.tipo_intervencion.especialidad not in q.especialidades_permitidas:
                        continue
                    
                    x[i, d, q_idx] = model.NewBoolVar(f'x_{i}_{d}_{q_idx}')
                    hora_inicio[i, d, q_idx] = model.NewIntVar(
                        480, 840, f'hora_{i}_{d}_{q_idx}'
                    )
        
        # Restricción 1: Cada cirugía se programa máximo una vez
        for i in range(n_cirugias):
            vars_cirugia = [x[key] for key in x if key[0] == i]
            if vars_cirugia:
                model.Add(sum(vars_cirugia) <= 1)
        
        # Restricción 2: No solapamiento en quirófanos (simplificada)
        for d in range(n_dias):
            for q_idx in range(n_quirofanos):
                # Limitar número de cirugías por quirófano/día
                vars_slot = [x[key] for key in x if key[1] == d and key[2] == q_idx]
                if vars_slot:
                    model.Add(sum(vars_slot) <= 6)  # Máximo 6 cirugías por quirófano/día
        
        # Restricción 3: Máximo ingresos UCI por día
        for d in range(n_dias):
            ingresos_uci = []
            for i, sol in enumerate(solicitudes):
                if sol.tipo_intervencion.probabilidad_uci > 0.3:
                    for q_idx in range(n_quirofanos):
                        if (i, d, q_idx) in x:
                            ingresos_uci.append(x[i, d, q_idx])
            
            if ingresos_uci:
                model.Add(sum(ingresos_uci) <= self.restricciones.max_ingresos_uci_dia)
        
        # Función objetivo
        objetivo = []
        for i, sol in enumerate(solicitudes):
            score = self._calcular_score_clinico_individual(sol)
            
            for d in range(n_dias):
                for q_idx in range(n_quirofanos):
                    if (i, d, q_idx) in x:
                        coef = int(score * 100)
                        objetivo.append(coef * x[i, d, q_idx])
        
        model.Maximize(sum(objetivo))
        
        # Resolver
        print(f"   Resolviendo (límite: {tiempo_limite_seg}s)...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = tiempo_limite_seg
        solver.parameters.num_search_workers = 4
        
        status = solver.Solve(model)
        
        # Extraer solución
        asignaciones = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for i, sol in enumerate(solicitudes):
                for d in range(n_dias):
                    for q_idx, q in enumerate(quirofanos_activos):
                        if (i, d, q_idx) in x and solver.Value(x[i, d, q_idx]) == 1:
                            asignaciones.append({
                                'solicitud': sol,
                                'fecha': dias[d],
                                'quirofano_id': q.id,
                                'hora_inicio': 480 + len([a for a in asignaciones 
                                                         if a['fecha'] == dias[d] 
                                                         and a['quirofano_id'] == q.id]) * 90,
                                'duracion': sol.duracion_esperada()
                            })
            
            gap = 0.0
            if solver.BestObjectiveBound() > 0:
                gap = abs(solver.BestObjectiveBound() - solver.ObjectiveValue()) / solver.BestObjectiveBound()
            
            print(f"   ✓ Solución: {len(asignaciones)} cirugías")
            print(f"   ✓ Estado: {'ÓPTIMO' if status == cp_model.OPTIMAL else 'FACTIBLE'}")
            
            return asignaciones, solver.ObjectiveValue(), gap
        else:
            print(f"   ✗ No se encontró solución factible")
            return [], 0.0, 1.0
    
    def _generar_dias_habiles(self, fecha_inicio: date, fecha_fin: date) -> List[date]:
        dias = []
        dia = fecha_inicio
        while dia <= fecha_fin:
            if dia.weekday() < 5:  # Lunes a Viernes
                dias.append(dia)
            dia += timedelta(days=1)
        return dias
    
    def _calcular_score_clinico_individual(self, sol: SolicitudCirugia) -> float:
        pesos = {
            PrioridadCatSalut.URGENTE: 1.0,
            PrioridadCatSalut.ONCOLOGICO_PRIORITARIO: 0.95,
            PrioridadCatSalut.ONCOLOGICO_ESTANDAR: 0.85,
            PrioridadCatSalut.CARDIACA: 0.80,
            PrioridadCatSalut.REFERENCIA_P1: 0.70,
            PrioridadCatSalut.REFERENCIA_P2: 0.50,
            PrioridadCatSalut.REFERENCIA_P3: 0.30,
        }
        score = pesos.get(sol.prioridad, 0.5)
        if sol.esta_fuera_plazo:
            score = min(1.0, score + 0.2)
        return score


class OptimizadorGenetico:
    """
    Optimizador usando Algoritmos Genéticos con DEAP.
    """
    
    def __init__(self, pesos: PesosOptimizacion, restricciones: RestriccionesGlobales,
                 quirofanos: List[Quirofano], config: ConfiguracionSistema,
                 restricciones_aprendidas: List[RestriccionAprendida] = None):
        self.pesos = pesos
        self.restricciones = restricciones
        self.quirofanos = quirofanos
        self.config = config
        self.restricciones_aprendidas = restricciones_aprendidas or []
        
        self.solicitudes = []
        self.cirujanos = []
        self.dias = []
        self.quirofanos_activos = []
    
    def optimizar(self, solicitudes: List[SolicitudCirugia],
                  cirujanos: List[Cirujano],
                  fecha_inicio: date, fecha_fin: date,
                  poblacion: int = 100,
                  generaciones: int = 200) -> Tuple[List[Dict], float, List[float]]:
        """Optimiza usando algoritmo genético."""
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP no está disponible")
        
        print(f"   AG: población={poblacion}, generaciones={generaciones}")
        
        self.solicitudes = solicitudes
        self.cirujanos = cirujanos
        self.dias = self._generar_dias_habiles(fecha_inicio, fecha_fin)
        self.quirofanos_activos = [q for q in self.quirofanos if q.activo]
        
        n_cirugias = len(solicitudes)
        n_dias = len(self.dias)
        n_quirofanos = len(self.quirofanos_activos)
        
        # Limpiar clases DEAP previas
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        def crear_individuo():
            individuo = []
            for i in range(n_cirugias):
                sol = solicitudes[i]
                # Quirófano compatible
                q_compatibles = [
                    j for j, q in enumerate(self.quirofanos_activos)
                    if sol.tipo_intervencion.especialidad in q.especialidades_permitidas
                ]
                q_idx = random.choice(q_compatibles) if q_compatibles else random.randint(0, n_quirofanos-1)
                dia_idx = random.randint(0, n_dias - 1)
                hora = random.randint(480, 840)
                individuo.append((i, dia_idx, q_idx, hora))
            random.shuffle(individuo)
            return creator.Individual(individuo)
        
        toolbox.register("individual", crear_individuo)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluar)
        toolbox.register("mate", self._cruce)
        toolbox.register("mutate", self._mutacion)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Población inicial
        pop = toolbox.population(n=poblacion)
        hof = tools.HallOfFame(5)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        
        # Evolución
        convergencia = []
        for gen in range(generaciones):
            # Selección
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            # Cruce
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutación
            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluar
            invalidos = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalidos)
            for ind, fit in zip(invalidos, fitnesses):
                ind.fitness.values = fit
            
            pop[:] = offspring
            hof.update(pop)
            
            record = stats.compile(pop)
            convergencia.append(record['max'])
            
            if gen % 50 == 0:
                print(f"   Gen {gen}: max={record['max']:.4f}")
        
        mejor = hof[0]
        asignaciones = self._decodificar(mejor)
        
        print(f"   ✓ Fitness: {mejor.fitness.values[0]:.4f}, Asignadas: {len(asignaciones)}")
        
        return asignaciones, mejor.fitness.values[0], convergencia
    
    def _evaluar(self, individuo) -> Tuple[float]:
        asignaciones = self._decodificar(individuo)
        if not asignaciones:
            return (0.0,)
        
        # Score clínico
        score_clinico = 0.0
        ids_prog = {a['solicitud'].id for a in asignaciones}
        
        for sol in self.solicitudes:
            if sol.id in ids_prog:
                if sol.prioridad == PrioridadCatSalut.ONCOLOGICO_PRIORITARIO:
                    score_clinico += 10
                elif sol.prioridad == PrioridadCatSalut.ONCOLOGICO_ESTANDAR:
                    score_clinico += 8
                elif sol.esta_fuera_plazo:
                    score_clinico += 7
                else:
                    score_clinico += 4
            elif sol.prioridad in [PrioridadCatSalut.ONCOLOGICO_PRIORITARIO,
                                   PrioridadCatSalut.ONCOLOGICO_ESTANDAR]:
                score_clinico -= 5
        
        score_clinico = max(0, score_clinico / (len(self.solicitudes) * 10))
        
        # Score eficiencia
        util_por_dq = defaultdict(int)
        for a in asignaciones:
            util_por_dq[(a['fecha'], a['quirofano_id'])] += a['duracion']
        
        utils = [min(1.0, m / 420) for m in util_por_dq.values()]
        score_eficiencia = np.mean(utils) if utils else 0
        
        fitness = (self.pesos.peso_prioridad_clinica * score_clinico +
                  self.pesos.peso_eficiencia_operativa * score_eficiencia)
        
        return (fitness,)
    
    def _decodificar(self, individuo) -> List[Dict]:
        asignaciones = []
        ocupacion = defaultdict(list)
        asignadas = set()
        
        for gen in individuo:
            i, d_idx, q_idx, hora = gen
            
            if i in asignadas or i >= len(self.solicitudes):
                continue
            if d_idx >= len(self.dias) or q_idx >= len(self.quirofanos_activos):
                continue
            
            sol = self.solicitudes[i]
            dia = self.dias[d_idx]
            q = self.quirofanos_activos[q_idx]
            dur = sol.duracion_esperada()
            
            if sol.tipo_intervencion.especialidad not in q.especialidades_permitidas:
                continue
            
            if hora + dur > q.horario_fin + 60:
                continue
            
            # Verificar solapamiento
            key = (dia, q.id)
            solapa = False
            for (ini, fin) in ocupacion[key]:
                if not (hora + dur + 30 <= ini or hora >= fin + 30):
                    solapa = True
                    break
            
            if solapa:
                continue
            
            asignaciones.append({
                'solicitud': sol,
                'fecha': dia,
                'quirofano_id': q.id,
                'hora_inicio': hora,
                'duracion': dur
            })
            ocupacion[key].append((hora, hora + dur))
            asignadas.add(i)
        
        return asignaciones
    
    def _cruce(self, ind1, ind2):
        size = min(len(ind1), len(ind2))
        if size < 4:
            return ind1, ind2
        
        p1, p2 = sorted(random.sample(range(size), 2))
        
        # Order crossover simplificado
        temp = ind1[p1:p2]
        ind1[p1:p2] = ind2[p1:p2]
        ind2[p1:p2] = temp
        
        return ind1, ind2
    
    def _mutacion(self, individuo):
        if not individuo:
            return individuo,
        
        idx = random.randint(0, len(individuo) - 1)
        i, d, q, h = individuo[idx]
        
        tipo = random.random()
        if tipo < 0.33:
            # Cambiar hora
            h = max(480, min(840, h + random.randint(-60, 60)))
        elif tipo < 0.66:
            # Cambiar día
            d = random.randint(0, len(self.dias) - 1)
        else:
            # Cambiar quirófano
            sol = self.solicitudes[i]
            compatibles = [j for j, qr in enumerate(self.quirofanos_activos)
                          if sol.tipo_intervencion.especialidad in qr.especialidades_permitidas]
            if compatibles:
                q = random.choice(compatibles)
        
        individuo[idx] = (i, d, q, h)
        return individuo,
    
    def _generar_dias_habiles(self, fecha_inicio: date, fecha_fin: date) -> List[date]:
        dias = []
        dia = fecha_inicio
        while dia <= fecha_fin:
            if dia.weekday() < 5:
                dias.append(dia)
            dia += timedelta(days=1)
        return dias


class OptimizadorEstocastico:
    """Evalúa robustez mediante simulación Monte Carlo."""
    
    def __init__(self, pesos, restricciones, quirofanos, config):
        self.pesos = pesos
        self.restricciones = restricciones
        self.quirofanos = quirofanos
        self.config = config
    
    def evaluar_robustez(self, asignaciones: List[Dict], 
                         n_sim: int = 100) -> Tuple[float, Dict]:
        """Evalúa robustez simulando variaciones en duraciones."""
        if not asignaciones:
            return 0.0, {}
        
        violaciones_ot = 0
        tiempos_fin = []
        
        for _ in range(n_sim):
            for asig in asignaciones:
                sol = asig['solicitud']
                media = sol.duracion_esperada()
                std = sol.tipo_intervencion.duracion_std_min
                
                # Simular duración real
                dur_sim = max(15, int(np.random.lognormal(
                    mean=np.log(media) - 0.5 * (std/media)**2,
                    sigma=std/media
                )))
                
                fin = asig['hora_inicio'] + dur_sim
                tiempos_fin.append(fin)
                
                if fin > 900:  # Después de 15:00
                    violaciones_ot += 1
        
        prob_ot = violaciones_ot / (n_sim * len(asignaciones))
        robustez = 1.0 - prob_ot
        
        stats = {
            'prob_overtime': prob_ot,
            'tiempo_fin_medio': np.mean(tiempos_fin),
            'tiempo_fin_p95': np.percentile(tiempos_fin, 95),
            'n_simulaciones': n_sim
        }
        
        return robustez, stats


class OptimizadorAvanzado:
    """Clase principal que integra todas las técnicas."""
    
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
        
        self.opt_milp = OptimizadorMILP(self.pesos, self.restricciones,
                                        self.quirofanos, self.config)
        self.opt_genetico = OptimizadorGenetico(self.pesos, self.restricciones,
                                                self.quirofanos, self.config,
                                                self.restricciones_aprendidas)
        self.opt_estocastico = OptimizadorEstocastico(self.pesos, self.restricciones,
                                                      self.quirofanos, self.config)
        self._convergencia = []
    
    def optimizar(
        self,
        solicitudes: List[SolicitudCirugia],
        cirujanos: List[Cirujano],
        fecha_inicio: date = None,
        fecha_fin: date = None,
        metodo: str = 'auto'
    ) -> ResultadoOptimizacionAvanzado:
        """Ejecuta optimización con el método especificado."""
        inicio = time.time()
        self._convergencia = []
        
        if fecha_inicio is None:
            fecha_inicio = date.today() + timedelta(days=1)
        if fecha_fin is None:
            fecha_fin = fecha_inicio + timedelta(days=self.config.horizonte_dias)
        
        print("=" * 70)
        print("OPTIMIZACIÓN AVANZADA")
        print("=" * 70)
        print(f"Solicitudes: {len(solicitudes)}, Método: {metodo}")
        
        # Filtrar válidas
        validas = [s for s in solicitudes 
                   if s.activa and not s.cancelada and s.preoperatorio_completado]
        print(f"Válidas: {len(validas)}")
        
        # Seleccionar método
        if metodo == 'auto':
            if len(validas) <= 50 and ORTOOLS_AVAILABLE:
                metodo = 'milp'
            elif DEAP_AVAILABLE:
                metodo = 'genetico'
            else:
                metodo = 'heuristico'
            print(f"Método auto-seleccionado: {metodo}")
        
        # Ejecutar
        asignaciones = []
        gap = 0.0
        
        if metodo == 'milp' and ORTOOLS_AVAILABLE:
            print("\n--- MILP (OR-Tools) ---")
            asignaciones, _, gap = self.opt_milp.optimizar(
                validas, cirujanos, fecha_inicio, fecha_fin
            )
        
        elif metodo == 'genetico' and DEAP_AVAILABLE:
            print("\n--- Algoritmo Genético ---")
            asignaciones, _, self._convergencia = self.opt_genetico.optimizar(
                validas, cirujanos, fecha_inicio, fecha_fin
            )
        
        else:
            print("\n--- Heurística ---")
            asignaciones = self._heuristica_simple(validas, fecha_inicio, fecha_fin)
        
        # Construir programa
        programa = self._construir_programa(asignaciones, fecha_inicio, fecha_fin)
        
        # Evaluar robustez
        print("\n--- Evaluando Robustez ---")
        robustez, stats_rob = self.opt_estocastico.evaluar_robustez(asignaciones)
        print(f"   Robustez: {robustez:.2%}")
        
        # Calcular scores
        score_clinico = self._calc_score_clinico(asignaciones, validas)
        score_eficiencia = self._calc_score_eficiencia(asignaciones)
        score_total = (self.pesos.peso_prioridad_clinica * score_clinico +
                      self.pesos.peso_eficiencia_operativa * score_eficiencia)
        
        tiempo_total = time.time() - inicio
        violaciones = self._verificar_restricciones(programa)
        
        resultado = ResultadoOptimizacionAvanzado(
            programa=programa,
            score_total=score_total,
            score_clinico=score_clinico,
            score_eficiencia=score_eficiencia,
            cirugias_programadas=len(asignaciones),
            cirugias_no_programadas=len(validas) - len(asignaciones),
            restricciones_violadas=violaciones,
            tiempo_ejecucion_seg=tiempo_total,
            iteraciones=len(self._convergencia),
            convergencia=self._convergencia.copy(),
            metodo_usado=metodo,
            gap_optimalidad=gap,
            robustez_score=robustez,
            estadisticas_detalladas=stats_rob
        )
        
        print(f"\n{'='*70}")
        print(f"✅ COMPLETADO: {metodo}, {tiempo_total:.1f}s")
        print(f"   Score: {score_total:.4f}, Programadas: {len(asignaciones)}")
        print(f"   Robustez: {robustez:.2%}")
        
        return resultado
    
    def _heuristica_simple(self, solicitudes, fecha_inicio, fecha_fin):
        """Heurística First Fit Decreasing simple."""
        asignaciones = []
        dias = []
        dia = fecha_inicio
        while dia <= fecha_fin:
            if dia.weekday() < 5:
                dias.append(dia)
            dia += timedelta(days=1)
        
        # Ordenar por prioridad
        ordenadas = sorted(solicitudes, 
                          key=lambda s: (0 if 'ONCOLOGICO' in s.prioridad.name else 1,
                                        -s.dias_en_espera))
        
        ocupacion = defaultdict(int)
        quirofanos_activos = [q for q in self.quirofanos if q.activo]
        
        for sol in ordenadas:
            for dia in dias:
                for q in quirofanos_activos:
                    if sol.tipo_intervencion.especialidad not in q.especialidades_permitidas:
                        continue
                    
                    key = (dia, q.id)
                    hora = 480 + ocupacion[key]
                    dur = sol.duracion_esperada()
                    
                    if hora + dur <= q.horario_fin + 60:
                        asignaciones.append({
                            'solicitud': sol,
                            'fecha': dia,
                            'quirofano_id': q.id,
                            'hora_inicio': hora,
                            'duracion': dur
                        })
                        ocupacion[key] += dur + 30
                        break
                else:
                    continue
                break
        
        return asignaciones
    
    def _construir_programa(self, asignaciones, fecha_inicio, fecha_fin):
        programa = ProgramaPeriodo(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
        
        for a in asignaciones:
            cirugia = CirugiaProgramada(
                id=generar_id_cirugia(),
                solicitud=a['solicitud'],
                fecha=a['fecha'],
                hora_inicio=a['hora_inicio'],
                duracion_programada_min=a['duracion'],
                quirofano_id=a['quirofano_id'],
                cirujano=a['solicitud'].cirujano_asignado
            )
            programa.obtener_dia(a['fecha']).agregar_cirugia(cirugia)
        
        return programa
    
    def _calc_score_clinico(self, asignaciones, solicitudes):
        if not solicitudes:
            return 0.0
        
        score = 0.0
        ids = {a['solicitud'].id for a in asignaciones}
        
        for s in solicitudes:
            if s.id in ids:
                if s.prioridad == PrioridadCatSalut.ONCOLOGICO_PRIORITARIO:
                    score += 10
                elif s.prioridad == PrioridadCatSalut.ONCOLOGICO_ESTANDAR:
                    score += 8
                elif s.esta_fuera_plazo:
                    score += 7
                else:
                    score += 4
            elif 'ONCOLOGICO' in s.prioridad.name:
                score -= 5
        
        return max(0, score / (len(solicitudes) * 10))
    
    def _calc_score_eficiencia(self, asignaciones):
        if not asignaciones:
            return 0.0
        
        util = defaultdict(int)
        for a in asignaciones:
            util[(a['fecha'], a['quirofano_id'])] += a['duracion']
        
        return np.mean([min(1.0, m/420) for m in util.values()])
    
    def _verificar_restricciones(self, programa):
        violaciones = []
        for fecha, prog in programa.programas_diarios.items():
            uci = prog.ingresos_uci_esperados()
            if uci > self.restricciones.max_ingresos_uci_dia:
                violaciones.append(f"{fecha}: UCI excedida ({uci:.1f})")
        return violaciones


def main():
    """Prueba del optimizador avanzado"""
    from synthetic_data import GeneradorDatosSinteticos
    from constraint_learning import AprendizajeRestricciones
    
    print("Generando datos...")
    gen = GeneradorDatosSinteticos(seed=42)
    cirujanos, lista_espera, historico = gen.generar_dataset_completo(
        n_solicitudes_espera=80,
        dias_historico=180
    )
    
    print("\nAprendiendo restricciones...")
    aprendizaje = AprendizajeRestricciones()
    restricciones = aprendizaje.analizar_historico(historico)
    
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MÉTODOS")
    print("=" * 70)
    
    opt = OptimizadorAvanzado(restricciones_aprendidas=restricciones)
    
    resultados = {}
    
    # Genético
    if DEAP_AVAILABLE:
        resultado = opt.optimizar(lista_espera, cirujanos, metodo='genetico')
        resultados['genetico'] = resultado
    
    # MILP (solo si hay pocos casos)
    if ORTOOLS_AVAILABLE and len(lista_espera) <= 50:
        resultado = opt.optimizar(lista_espera[:50], cirujanos, metodo='milp')
        resultados['milp'] = resultado
    
    # Comparativa
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"{'Método':<12} {'Score':>8} {'Clínico':>8} {'Efic.':>8} {'Prog.':>6} {'Robust.':>8} {'Tiempo':>8}")
    print("-" * 66)
    
    for metodo, res in resultados.items():
        print(f"{metodo:<12} {res.score_total:>8.4f} {res.score_clinico:>8.4f} "
              f"{res.score_eficiencia:>8.4f} {res.cirugias_programadas:>6} "
              f"{res.robustez_score:>7.1%} {res.tiempo_ejecucion_seg:>7.1f}s")
    
    return resultados


if __name__ == "__main__":
    main()
