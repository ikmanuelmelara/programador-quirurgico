"""
MÃ³dulo de Aprendizaje AutomÃ¡tico de Restricciones
=================================================
Analiza datos histÃ³ricos para descubrir restricciones implÃ­citas
y patrones que el modelo de optimizaciÃ³n debe considerar.

Basado en la literatura:
- Association Rules Mining para patrones frecuentes
- AnÃ¡lisis de secuencias temporales
- DetecciÃ³n de preferencias de cirujanos/quirÃ³fanos
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass, field
import warnings

from config import (
    ConfigAprendizaje, CONFIG_APRENDIZAJE_DEFAULT,
    Especialidad, QUIROFANOS_DEFAULT
)
from models import RestriccionAprendida, generar_id


@dataclass
class PatronDescubierto:
    """Representa un patrÃ³n descubierto en los datos histÃ³ricos"""
    tipo: str
    descripcion: str
    entidades: Dict[str, Any]
    soporte: float
    confianza: float
    lift: float = 1.0
    ejemplos: int = 0


class AprendizajeRestricciones:
    """
    Sistema de aprendizaje automÃ¡tico de restricciones basado en datos histÃ³ricos.
    
    Detecta:
    1. Preferencias de cirujanos por quirÃ³fanos
    2. Patrones de secuenciaciÃ³n de cirugÃ­as
    3. Restricciones de tiempo entre procedimientos
    4. Co-ocurrencias frecuentes
    5. AnomalÃ­as y reglas implÃ­citas
    """
    
    def __init__(self, config: ConfigAprendizaje = None):
        self.config = config or CONFIG_APRENDIZAJE_DEFAULT
        self.restricciones_aprendidas: List[RestriccionAprendida] = []
        self.patrones_descubiertos: List[PatronDescubierto] = []
        self._estadisticas_base: Dict[str, Any] = {}
    
    def analizar_historico(self, historico_df: pd.DataFrame) -> List[RestriccionAprendida]:
        """
        Analiza el histÃ³rico completo y descubre restricciones.
        
        Args:
            historico_df: DataFrame con el histÃ³rico de cirugÃ­as
        
        Returns:
            Lista de restricciones aprendidas
        """
        if len(historico_df) < self.config.min_casos_para_aprender:
            warnings.warn(
                f"HistÃ³rico insuficiente: {len(historico_df)} casos "
                f"(mÃ­nimo: {self.config.min_casos_para_aprender})"
            )
            return []
        
        print("=" * 60)
        print("APRENDIZAJE AUTOMÃTICO DE RESTRICCIONES")
        print("=" * 60)
        print(f"Analizando {len(historico_df)} registros histÃ³ricos...")
        
        # Calcular estadÃ­sticas base
        self._calcular_estadisticas_base(historico_df)
        
        # Ejecutar anÃ¡lisis
        restricciones = []
        
        # 1. Preferencias de cirujano-quirÃ³fano
        print("\n1. Analizando preferencias cirujano-quirÃ³fano...")
        restricciones.extend(self._aprender_preferencias_cirujano_quirofano(historico_df))
        
        # 2. Patrones de secuenciaciÃ³n
        print("2. Analizando patrones de secuenciaciÃ³n...")
        restricciones.extend(self._aprender_patrones_secuencia(historico_df))
        
        # 3. Restricciones temporales
        print("3. Analizando restricciones temporales...")
        restricciones.extend(self._aprender_restricciones_temporales(historico_df))
        
        # 4. Patrones de duraciÃ³n
        print("4. Analizando patrones de duraciÃ³n...")
        restricciones.extend(self._aprender_patrones_duracion(historico_df))
        
        # 5. Reglas de especialidad-quirÃ³fano
        print("5. Analizando asignaciÃ³n especialidad-quirÃ³fano...")
        restricciones.extend(self._aprender_asignacion_especialidad(historico_df))
        
        # 6. Patrones de dÃ­a de semana
        print("6. Analizando patrones por dÃ­a de semana...")
        restricciones.extend(self._aprender_patrones_dia_semana(historico_df))
        
        self.restricciones_aprendidas = restricciones
        
        print(f"\n=> Total restricciones descubiertas: {len(restricciones)}")
        
        return restricciones
    
    def _calcular_estadisticas_base(self, df: pd.DataFrame):
        """Calcula estadÃ­sticas bÃ¡sicas del histÃ³rico"""
        self._estadisticas_base = {
            'total_cirugias': len(df),
            'dias_analizados': df['fecha'].nunique(),
            'cirujanos_unicos': df['cirujano_id'].nunique(),
            'quirofanos_usados': df['quirofano_id'].nunique(),
            'tipos_intervencion': df['tipo_intervencion'].nunique(),
            'duracion_media_global': df['duracion_real'].mean(),
            'duracion_std_global': df['duracion_real'].std(),
            'overtime_medio': df['overtime'].mean(),
            'tasa_complicaciones': df['complicacion'].mean(),
        }
    
    def _aprender_preferencias_cirujano_quirofano(
        self, df: pd.DataFrame
    ) -> List[RestriccionAprendida]:
        """
        Descubre preferencias de cirujanos por quirÃ³fanos especÃ­ficos.
        Detecta cuando un cirujano opera consistentemente en el mismo quirÃ³fano.
        """
        restricciones = []
        
        # Agrupar por cirujano
        for cirujano_id in df['cirujano_id'].unique():
            if pd.isna(cirujano_id):
                continue
                
            df_cirujano = df[df['cirujano_id'] == cirujano_id]
            total_cirugias = len(df_cirujano)
            
            if total_cirugias < 20:  # MÃ­nimo para detectar patrÃ³n
                continue
            
            # Contar uso de quirÃ³fanos
            conteo_quirofanos = df_cirujano['quirofano_id'].value_counts()
            
            for quirofano_id, count in conteo_quirofanos.items():
                proporcion = count / total_cirugias
                
                if proporcion >= self.config.umbral_preferencia_quirofano:
                    restriccion = RestriccionAprendida(
                        id=generar_id(),
                        tipo='preferencia_quirofano',
                        descripcion=f"El cirujano {cirujano_id} prefiere operar en quirÃ³fano {quirofano_id} ({proporcion*100:.1f}% de sus cirugÃ­as)",
                        entidades={
                            'cirujano_id': cirujano_id,
                            'quirofano_id': quirofano_id,
                        },
                        soporte=count / len(df),
                        confianza=proporcion,
                        lift=proporcion / (df['quirofano_id'] == quirofano_id).mean(),
                        penalizacion_incumplimiento=0.5,
                        es_hard_constraint=False
                    )
                    restricciones.append(restriccion)
        
        print(f"   -> {len(restricciones)} preferencias cirujano-quirÃ³fano detectadas")
        return restricciones
    
    def _aprender_patrones_secuencia(
        self, df: pd.DataFrame
    ) -> List[RestriccionAprendida]:
        """
        Descubre patrones de secuenciaciÃ³n de tipos de cirugÃ­a.
        Por ejemplo: "DespuÃ©s de cirugÃ­a oncolÃ³gica compleja, no programar otra igual el mismo dÃ­a"
        """
        restricciones = []
        
        # Agrupar por dÃ­a y quirÃ³fano
        for (fecha, quirofano), grupo in df.groupby(['fecha', 'quirofano_id']):
            grupo = grupo.sort_values('hora_inicio')
            if len(grupo) < 2:
                continue
            
            # Analizar secuencias de 2 cirugÃ­as consecutivas
            for i in range(len(grupo) - 1):
                tipo1 = grupo.iloc[i]['tipo_intervencion']
                tipo2 = grupo.iloc[i + 1]['tipo_intervencion']
                # Almacenar para anÃ¡lisis posterior
        
        # Patrones pre-definidos basados en literatura
        patrones_literatura = [
            {
                'tipo': 'secuencia_oncologico',
                'descripcion': 'Las cirugÃ­as oncolÃ³gicas complejas se programan a primera hora',
                'entidades': {'tipos': ['COLECT_DER', 'COLECT_IZQ', 'RECT_ANT', 'GASTRECT', 'WHIPPLE']},
                'soporte': 0.15,
                'confianza': 0.8,
            },
            {
                'tipo': 'secuencia_contaminada',
                'descripcion': 'CirugÃ­as potencialmente contaminadas se programan al final del dÃ­a',
                'entidades': {'tipos': ['FISTULA_AN', 'HEMORR', 'PILONIDAL']},
                'soporte': 0.1,
                'confianza': 0.75,
            },
        ]
        
        for patron in patrones_literatura:
            # Verificar si el patrÃ³n se cumple en los datos
            tipos_relevantes = patron['entidades']['tipos']
            df_relevante = df[df['tipo_intervencion'].isin(tipos_relevantes)]
            
            if len(df_relevante) < 10:
                continue
            
            if patron['tipo'] == 'secuencia_oncologico':
                # Verificar si estÃ¡n a primera hora
                primera_hora = df_relevante[df_relevante['hora_inicio'] < 9*60]
                conf_real = len(primera_hora) / len(df_relevante)
            else:
                # Verificar si estÃ¡n al final
                ultima_hora = df_relevante[df_relevante['hora_inicio'] > 13*60]
                conf_real = len(ultima_hora) / len(df_relevante) if len(df_relevante) > 0 else 0
            
            if conf_real >= 0.5:  # Umbral mÃ¡s bajo para validar
                restriccion = RestriccionAprendida(
                    id=generar_id(),
                    tipo=patron['tipo'],
                    descripcion=patron['descripcion'],
                    entidades=patron['entidades'],
                    soporte=len(df_relevante) / len(df),
                    confianza=conf_real,
                    penalizacion_incumplimiento=0.3,
                    es_hard_constraint=False
                )
                restricciones.append(restriccion)
        
        print(f"   -> {len(restricciones)} patrones de secuencia detectados")
        return restricciones
    
    def _aprender_restricciones_temporales(
        self, df: pd.DataFrame
    ) -> List[RestriccionAprendida]:
        """
        Descubre restricciones temporales como tiempos entre cirugÃ­as,
        horarios preferidos, etc.
        """
        restricciones = []
        
        # Analizar gaps entre cirugÃ­as por quirÃ³fano/dÃ­a
        gaps = []
        for (fecha, quirofano), grupo in df.groupby(['fecha', 'quirofano_id']):
            grupo = grupo.sort_values('hora_inicio')
            if len(grupo) < 2:
                continue
            
            for i in range(len(grupo) - 1):
                fin_actual = grupo.iloc[i]['hora_fin']
                inicio_sig = grupo.iloc[i + 1]['hora_inicio']
                gap = inicio_sig - fin_actual
                tipo_actual = grupo.iloc[i]['tipo_intervencion']
                gaps.append({
                    'gap': gap,
                    'tipo_previo': tipo_actual,
                    'quirofano': quirofano
                })
        
        if gaps:
            df_gaps = pd.DataFrame(gaps)
            gap_medio = df_gaps['gap'].mean()
            gap_std = df_gaps['gap'].std()
            
            # RestricciÃ³n de tiempo mÃ­nimo entre cirugÃ­as
            if gap_medio > 25:  # Si consistentemente dejan mÃ¡s de 25 min
                restriccion = RestriccionAprendida(
                    id=generar_id(),
                    tipo='tiempo_minimo_entre_cirugias',
                    descripcion=f"Tiempo medio entre cirugÃ­as: {gap_medio:.0f} min (std: {gap_std:.0f})",
                    entidades={
                        'tiempo_minimo': int(gap_medio - gap_std),
                        'tiempo_medio': int(gap_medio),
                    },
                    soporte=len(gaps) / len(df),
                    confianza=0.9,
                    penalizacion_incumplimiento=0.4,
                    es_hard_constraint=False
                )
                restricciones.append(restriccion)
        
        # AnÃ¡lisis de horas de inicio preferidas
        df['hora_inicio_bucket'] = (df['hora_inicio'] // 60).astype(int)
        dist_horas = df['hora_inicio_bucket'].value_counts(normalize=True)
        
        if dist_horas.iloc[0] > 0.3:  # Si hay una hora muy preferida
            hora_preferida = dist_horas.index[0]
            restriccion = RestriccionAprendida(
                id=generar_id(),
                tipo='hora_inicio_preferida',
                descripcion=f"Alta concentraciÃ³n de inicios a las {hora_preferida}:00 ({dist_horas.iloc[0]*100:.1f}%)",
                entidades={'hora_preferida': hora_preferida},
                soporte=dist_horas.iloc[0],
                confianza=dist_horas.iloc[0],
                penalizacion_incumplimiento=0.1,
                es_hard_constraint=False
            )
            restricciones.append(restriccion)
        
        print(f"   -> {len(restricciones)} restricciones temporales detectadas")
        return restricciones
    
    def _aprender_patrones_duracion(
        self, df: pd.DataFrame
    ) -> List[RestriccionAprendida]:
        """
        Aprende patrones de duraciÃ³n real vs programada para mejorar estimaciones.
        """
        restricciones = []
        
        # Calcular ratio duraciÃ³n real/programada por tipo
        df['ratio_duracion'] = df['duracion_real'] / df['duracion_programada']
        
        for tipo in df['tipo_intervencion'].unique():
            df_tipo = df[df['tipo_intervencion'] == tipo]
            
            if len(df_tipo) < 10:
                continue
            
            ratio_medio = df_tipo['ratio_duracion'].mean()
            ratio_std = df_tipo['ratio_duracion'].std()
            
            # Si consistentemente dura mÃ¡s o menos de lo programado
            if abs(ratio_medio - 1.0) > 0.15:  # MÃ¡s de 15% de desviaciÃ³n
                restriccion = RestriccionAprendida(
                    id=generar_id(),
                    tipo='ajuste_duracion',
                    descripcion=f"IntervenciÃ³n {tipo}: duraciÃ³n real = {ratio_medio:.2f}x programada (Â±{ratio_std:.2f})",
                    entidades={
                        'tipo_intervencion': tipo,
                        'factor_ajuste': ratio_medio,
                        'desviacion': ratio_std,
                    },
                    soporte=len(df_tipo) / len(df),
                    confianza=1 - ratio_std,  # MÃ¡s confiable si menos variabilidad
                    penalizacion_incumplimiento=0.0,  # Solo informativo
                    es_hard_constraint=False
                )
                restricciones.append(restriccion)
        
        print(f"   -> {len(restricciones)} ajustes de duraciÃ³n detectados")
        return restricciones
    
    def _aprender_asignacion_especialidad(
        self, df: pd.DataFrame
    ) -> List[RestriccionAprendida]:
        """
        Aprende quÃ© especialidades se asignan tÃ­picamente a quÃ© quirÃ³fanos.
        """
        restricciones = []
        
        # Matriz especialidad x quirÃ³fano
        matriz = pd.crosstab(
            df['especialidad'], 
            df['quirofano_id'], 
            normalize='index'
        )
        
        for esp in matriz.index:
            for quirofano in matriz.columns:
                proporcion = matriz.loc[esp, quirofano]
                
                # Si una especialidad usa principalmente un quirÃ³fano
                if proporcion >= 0.6:
                    # Calcular lift
                    prob_quirofano = (df['quirofano_id'] == quirofano).mean()
                    lift = proporcion / prob_quirofano if prob_quirofano > 0 else 1
                    
                    restriccion = RestriccionAprendida(
                        id=generar_id(),
                        tipo='asignacion_especialidad',
                        descripcion=f"Especialidad {esp} opera principalmente en quirÃ³fano {quirofano} ({proporcion*100:.1f}%)",
                        entidades={
                            'especialidad': esp,
                            'quirofano_id': quirofano,
                        },
                        soporte=(df['especialidad'] == esp).sum() / len(df),
                        confianza=proporcion,
                        lift=lift,
                        penalizacion_incumplimiento=0.3,
                        es_hard_constraint=False
                    )
                    restricciones.append(restriccion)
        
        print(f"   -> {len(restricciones)} asignaciones especialidad-quirÃ³fano detectadas")
        return restricciones
    
    def _aprender_patrones_dia_semana(
        self, df: pd.DataFrame
    ) -> List[RestriccionAprendida]:
        """
        Detecta patrones segÃºn el dÃ­a de la semana.
        """
        restricciones = []
        dias = ['Lunes', 'Martes', 'MiÃ©rcoles', 'Jueves', 'Viernes']
        
        # CirugÃ­as complejas por dÃ­a
        df['es_compleja'] = df['complejidad'] >= 4
        
        for dia_num in range(5):
            df_dia = df[df['dia_semana'] == dia_num]
            if len(df_dia) < 50:
                continue
            
            prop_complejas = df_dia['es_compleja'].mean()
            prop_global = df['es_compleja'].mean()
            
            # Si un dÃ­a tiene significativamente mÃ¡s o menos complejas
            if abs(prop_complejas - prop_global) > 0.1:
                restriccion = RestriccionAprendida(
                    id=generar_id(),
                    tipo='patron_dia_semana',
                    descripcion=f"{dias[dia_num]}: {'mÃ¡s' if prop_complejas > prop_global else 'menos'} cirugÃ­as complejas ({prop_complejas*100:.1f}% vs {prop_global*100:.1f}% global)",
                    entidades={
                        'dia_semana': dia_num,
                        'proporcion_complejas': prop_complejas,
                    },
                    soporte=len(df_dia) / len(df),
                    confianza=0.7,
                    penalizacion_incumplimiento=0.1,
                    es_hard_constraint=False
                )
                restricciones.append(restriccion)
        
        print(f"   -> {len(restricciones)} patrones de dÃ­a de semana detectados")
        return restricciones
    
    def obtener_restricciones_para_cirugia(
        self, 
        tipo_intervencion: str,
        cirujano_id: str = None,
        especialidad: str = None
    ) -> List[RestriccionAprendida]:
        """
        Obtiene las restricciones aprendidas relevantes para una cirugÃ­a especÃ­fica.
        """
        relevantes = []
        
        for r in self.restricciones_aprendidas:
            # Filtrar por tipo de intervenciÃ³n
            if 'tipo_intervencion' in r.entidades:
                if r.entidades['tipo_intervencion'] == tipo_intervencion:
                    relevantes.append(r)
                continue
            
            # Filtrar por cirujano
            if cirujano_id and 'cirujano_id' in r.entidades:
                if r.entidades['cirujano_id'] == cirujano_id:
                    relevantes.append(r)
                continue
            
            # Filtrar por especialidad
            if especialidad and 'especialidad' in r.entidades:
                if r.entidades['especialidad'] == especialidad:
                    relevantes.append(r)
                continue
            
            # Restricciones genÃ©ricas (sin filtro especÃ­fico de entidad)
            if r.tipo in ['tiempo_minimo_entre_cirugias', 'hora_inicio_preferida']:
                relevantes.append(r)
        
        return relevantes
    
    def generar_resumen(self) -> str:
        """Genera un resumen textual de las restricciones aprendidas."""
        if not self.restricciones_aprendidas:
            return "No se han aprendido restricciones todavÃ­a."
        
        lineas = [
            "=" * 60,
            "RESUMEN DE RESTRICCIONES APRENDIDAS",
            "=" * 60,
            f"Total de restricciones: {len(self.restricciones_aprendidas)}",
            ""
        ]
        
        # Agrupar por tipo
        por_tipo = defaultdict(list)
        for r in self.restricciones_aprendidas:
            por_tipo[r.tipo].append(r)
        
        for tipo, restricciones in por_tipo.items():
            lineas.append(f"\n[{tipo.upper()}] ({len(restricciones)} restricciones)")
            for r in restricciones[:5]:  # Mostrar mÃ¡ximo 5 por tipo
                lineas.append(f"  â€¢ {r.descripcion}")
                lineas.append(f"    (soporte: {r.soporte:.2%}, confianza: {r.confianza:.2%})")
            if len(restricciones) > 5:
                lineas.append(f"  ... y {len(restricciones) - 5} mÃ¡s")
        
        return "\n".join(lineas)
    
    def exportar_restricciones(self) -> pd.DataFrame:
        """Exporta las restricciones a un DataFrame para anÃ¡lisis."""
        datos = []
        for r in self.restricciones_aprendidas:
            datos.append({
                'id': r.id,
                'tipo': r.tipo,
                'descripcion': r.descripcion,
                'soporte': r.soporte,
                'confianza': r.confianza,
                'lift': r.lift,
                'penalizacion': r.penalizacion_incumplimiento,
                'es_hard': r.es_hard_constraint,
                'activa': r.activa,
            })
        return pd.DataFrame(datos)


def main():
    """Prueba del mÃ³dulo de aprendizaje"""
    from synthetic_data import GeneradorDatosSinteticos
    
    print("Generando datos sintÃ©ticos...")
    generador = GeneradorDatosSinteticos(seed=42)
    cirujanos, _, historico = generador.generar_dataset_completo(
        n_solicitudes_espera=100,
        dias_historico=365,
        cirugias_dia_historico=25
    )
    
    print("\n" + "=" * 60)
    aprendizaje = AprendizajeRestricciones()
    restricciones = aprendizaje.analizar_historico(historico)
    
    print("\n" + aprendizaje.generar_resumen())
    
    return aprendizaje


if __name__ == "__main__":
    main()
