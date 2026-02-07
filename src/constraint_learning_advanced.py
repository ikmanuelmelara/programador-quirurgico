"""
Módulo Avanzado de Aprendizaje de Restricciones
===============================================
Implementa técnicas reales de Machine Learning:
- Association Rules Mining (Apriori/FP-Growth)
- Clustering (K-Means, DBSCAN)
- Detección de patrones multi-variable
- Descubrimiento automático de nuevos tipos de restricciones

Dependencias: pip install mlxtend scikit-learn scipy
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass, field
import warnings
from itertools import combinations

# ML Libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from scipy import stats

try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("⚠️ mlxtend no disponible. Instalar con: pip install mlxtend")

from config import (
    ConfigAprendizaje, CONFIG_APRENDIZAJE_DEFAULT,
    Especialidad, QUIROFANOS_DEFAULT
)
from models import RestriccionAprendida, generar_id


@dataclass
class PatronDescubierto:
    """Representa un patrón descubierto automáticamente"""
    tipo: str
    descripcion: str
    variables_involucradas: List[str]
    regla: Dict[str, Any]
    soporte: float
    confianza: float
    lift: float = 1.0
    ejemplos: int = 0
    metodo_descubrimiento: str = ""  # 'apriori', 'clustering', 'decision_tree', etc.


@dataclass 
class ClusterDescubierto:
    """Representa un cluster de cirugías similares"""
    id: int
    centroide: Dict[str, float]
    tamano: int
    caracteristicas_distintivas: List[str]
    cirugias_tipicas: List[str]


class AprendizajeRestriccionesAvanzado:
    """
    Sistema avanzado de aprendizaje de restricciones usando ML real.
    
    Técnicas implementadas:
    1. Association Rules Mining (Apriori/FP-Growth)
    2. Clustering (K-Means, DBSCAN) 
    3. Árboles de decisión para reglas interpretables
    4. Detección de anomalías (Isolation Forest)
    5. Análisis de correlaciones multi-variable
    6. Descubrimiento automático de patrones temporales
    """
    
    def __init__(self, config: ConfigAprendizaje = None):
        self.config = config or CONFIG_APRENDIZAJE_DEFAULT
        self.restricciones_aprendidas: List[RestriccionAprendida] = []
        self.patrones_descubiertos: List[PatronDescubierto] = []
        self.clusters_descubiertos: List[ClusterDescubierto] = []
        self._estadisticas_base: Dict[str, Any] = {}
        self._encoders: Dict[str, LabelEncoder] = {}
        self._scaler = StandardScaler()
        
    def analizar_historico(self, historico_df: pd.DataFrame) -> List[RestriccionAprendida]:
        """
        Análisis completo del histórico usando múltiples técnicas de ML.
        """
        if len(historico_df) < 100:
            warnings.warn(f"Histórico pequeño: {len(historico_df)} casos. Resultados pueden ser poco fiables.")
        
        print("=" * 70)
        print("APRENDIZAJE AVANZADO DE RESTRICCIONES (ML)")
        print("=" * 70)
        print(f"Analizando {len(historico_df)} registros históricos...")
        
        # Preparar datos
        df = self._preparar_datos(historico_df)
        
        restricciones = []
        
        # 1. ASSOCIATION RULES MINING
        print("\n📊 1. ASSOCIATION RULES MINING (Apriori/FP-Growth)")
        print("-" * 50)
        restricciones.extend(self._association_rules_mining(df))
        
        # 2. CLUSTERING
        print("\n📊 2. CLUSTERING (K-Means + DBSCAN)")
        print("-" * 50)
        restricciones.extend(self._clustering_analysis(df))
        
        # 3. ÁRBOLES DE DECISIÓN
        print("\n📊 3. ÁRBOLES DE DECISIÓN (Reglas interpretables)")
        print("-" * 50)
        restricciones.extend(self._decision_tree_rules(df))
        
        # 4. DETECCIÓN DE ANOMALÍAS
        print("\n📊 4. DETECCIÓN DE ANOMALÍAS (Isolation Forest)")
        print("-" * 50)
        restricciones.extend(self._anomaly_detection(df))
        
        # 5. CORRELACIONES MULTI-VARIABLE
        print("\n📊 5. ANÁLISIS DE CORRELACIONES MULTI-VARIABLE")
        print("-" * 50)
        restricciones.extend(self._multivariate_correlations(df))
        
        # 6. PATRONES TEMPORALES AVANZADOS
        print("\n📊 6. PATRONES TEMPORALES AVANZADOS")
        print("-" * 50)
        restricciones.extend(self._temporal_pattern_mining(df))
        
        # 7. PATRONES DE SECUENCIA
        print("\n📊 7. SEQUENTIAL PATTERN MINING")
        print("-" * 50)
        restricciones.extend(self._sequential_pattern_mining(df))
        
        self.restricciones_aprendidas = restricciones
        
        print(f"\n{'='*70}")
        print(f"✅ TOTAL RESTRICCIONES DESCUBIERTAS: {len(restricciones)}")
        print(f"{'='*70}")
        
        return restricciones
    
    def _preparar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara y enriquece los datos para análisis"""
        df = df.copy()
        
        # Convertir fecha a datetime si no lo es
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
            df['dia_semana'] = df['fecha'].dt.dayofweek
            df['semana_mes'] = df['fecha'].dt.isocalendar().week
            df['mes'] = df['fecha'].dt.month
        
        # Crear bins para variables continuas
        if 'hora_inicio' in df.columns:
            df['franja_horaria'] = pd.cut(
                df['hora_inicio'], 
                bins=[0, 540, 660, 780, 900, 1440],  # 9:00, 11:00, 13:00, 15:00
                labels=['primera_hora', 'media_manana', 'mediodia', 'tarde', 'noche']
            )
        
        if 'duracion_real' in df.columns:
            df['duracion_categoria'] = pd.cut(
                df['duracion_real'],
                bins=[0, 60, 120, 180, 300, 1000],
                labels=['muy_corta', 'corta', 'media', 'larga', 'muy_larga']
            )
        
        if 'paciente_edad' in df.columns:
            df['grupo_edad'] = pd.cut(
                df['paciente_edad'],
                bins=[0, 40, 60, 75, 100],
                labels=['joven', 'adulto', 'mayor', 'anciano']
            )
        
        # Calcular ratio duración real/programada
        if 'duracion_real' in df.columns and 'duracion_programada' in df.columns:
            df['ratio_duracion'] = df['duracion_real'] / df['duracion_programada'].clip(lower=1)
            df['desviacion_duracion'] = df['duracion_real'] - df['duracion_programada']
        
        # Flag de overtime
        if 'hora_fin' in df.columns:
            df['tiene_overtime'] = df['hora_fin'] > 900  # Después de las 15:00
        
        return df
    
    # =========================================================================
    # 1. ASSOCIATION RULES MINING
    # =========================================================================
    
    def _association_rules_mining(self, df: pd.DataFrame) -> List[RestriccionAprendida]:
        """
        Descubre reglas de asociación usando Apriori o FP-Growth.
        Encuentra patrones como: {cirujano=X, día=Lunes} → {quirófano=2}
        """
        restricciones = []
        
        if not MLXTEND_AVAILABLE:
            print("   ⚠️ mlxtend no disponible, saltando Association Rules")
            return restricciones
        
        # Crear transacciones (cada cirugía es una transacción)
        columnas_categoricas = [
            'cirujano_id', 'quirofano_id', 'tipo_intervencion', 
            'especialidad', 'dia_semana', 'franja_horaria',
            'duracion_categoria', 'grupo_edad', 'prioridad'
        ]
        
        columnas_disponibles = [c for c in columnas_categoricas if c in df.columns]
        
        if len(columnas_disponibles) < 2:
            print("   ⚠️ Insuficientes columnas categóricas")
            return restricciones
        
        # Convertir a formato de transacciones
        transacciones = []
        for _, row in df.iterrows():
            items = []
            for col in columnas_disponibles:
                if pd.notna(row[col]):
                    items.append(f"{col}={row[col]}")
            transacciones.append(items)
        
        # Codificar transacciones
        te = TransactionEncoder()
        te_array = te.fit_transform(transacciones)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        
        # Aplicar FP-Growth (más eficiente que Apriori)
        try:
            frequent_itemsets = fpgrowth(
                df_encoded, 
                min_support=0.05,  # Mínimo 5% de soporte
                use_colnames=True
            )
            
            if len(frequent_itemsets) == 0:
                print("   No se encontraron itemsets frecuentes")
                return restricciones
            
            # Generar reglas de asociación
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence",
                min_threshold=0.6  # Mínimo 60% confianza
            )
            
            # Filtrar reglas interesantes (lift > 1.5)
            rules = rules[rules['lift'] > 1.5]
            
            print(f"   → {len(rules)} reglas de asociación descubiertas")
            
            # Convertir reglas a restricciones
            for _, rule in rules.head(20).iterrows():  # Top 20 reglas
                antecedentes = list(rule['antecedents'])
                consecuentes = list(rule['consequents'])
                
                descripcion = f"SI {' Y '.join(antecedentes)} → {' Y '.join(consecuentes)}"
                
                restriccion = RestriccionAprendida(
                    id=generar_id(),
                    tipo='regla_asociacion',
                    descripcion=descripcion[:200],
                    entidades={
                        'antecedentes': antecedentes,
                        'consecuentes': consecuentes,
                    },
                    soporte=rule['support'],
                    confianza=rule['confidence'],
                    lift=rule['lift'],
                    penalizacion_incumplimiento=min(0.5, rule['lift'] / 10),
                    es_hard_constraint=False
                )
                restricciones.append(restriccion)
                
                self.patrones_descubiertos.append(PatronDescubierto(
                    tipo='association_rule',
                    descripcion=descripcion,
                    variables_involucradas=antecedentes + consecuentes,
                    regla={'antecedents': antecedentes, 'consequents': consecuentes},
                    soporte=rule['support'],
                    confianza=rule['confidence'],
                    lift=rule['lift'],
                    metodo_descubrimiento='fp_growth'
                ))
                
        except Exception as e:
            print(f"   ⚠️ Error en Association Rules: {e}")
        
        return restricciones
    
    # =========================================================================
    # 2. CLUSTERING
    # =========================================================================
    
    def _clustering_analysis(self, df: pd.DataFrame) -> List[RestriccionAprendida]:
        """
        Descubre grupos de cirugías similares usando clustering.
        Identifica patrones como: "Cirugías tipo X se agrupan en quirófano Y por las mañanas"
        """
        restricciones = []
        
        # Seleccionar features numéricas y codificar categóricas
        features_numericas = ['hora_inicio', 'duracion_real', 'quirofano_id', 
                             'dia_semana', 'complejidad', 'paciente_edad']
        features_disponibles = [f for f in features_numericas if f in df.columns]
        
        if len(features_disponibles) < 3:
            print("   ⚠️ Insuficientes features para clustering")
            return restricciones
        
        # Preparar matriz de features
        X = df[features_disponibles].dropna()
        
        if len(X) < 50:
            print("   ⚠️ Insuficientes datos para clustering")
            return restricciones
        
        # Normalizar
        X_scaled = self._scaler.fit_transform(X)
        
        # K-Means con selección automática de K (método del codo simplificado)
        best_k = self._find_optimal_k(X_scaled, max_k=8)
        
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        print(f"   → K-Means: {best_k} clusters óptimos encontrados")
        
        # Analizar cada cluster
        df_clustered = X.copy()
        df_clustered['cluster'] = clusters
        
        # Añadir columnas categóricas para análisis
        for col in ['tipo_intervencion', 'especialidad', 'cirujano_id']:
            if col in df.columns:
                df_clustered[col] = df.loc[X.index, col].values
        
        for cluster_id in range(best_k):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            if cluster_size < 10:
                continue
            
            # Encontrar características distintivas del cluster
            caracteristicas = self._caracterizar_cluster(cluster_data, df_clustered)
            
            if caracteristicas:
                descripcion = f"Cluster {cluster_id}: {', '.join(caracteristicas[:3])}"
                
                restriccion = RestriccionAprendida(
                    id=generar_id(),
                    tipo='cluster_cirugia',
                    descripcion=descripcion,
                    entidades={
                        'cluster_id': cluster_id,
                        'tamano': cluster_size,
                        'caracteristicas': caracteristicas,
                        'centroide': dict(zip(features_disponibles, kmeans.cluster_centers_[cluster_id]))
                    },
                    soporte=cluster_size / len(df),
                    confianza=0.8,
                    penalizacion_incumplimiento=0.2,
                    es_hard_constraint=False
                )
                restricciones.append(restriccion)
        
        # DBSCAN para detectar outliers y clusters de forma irregular
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        dbscan_clusters = dbscan.fit_predict(X_scaled)
        
        n_outliers = (dbscan_clusters == -1).sum()
        n_clusters_dbscan = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
        
        print(f"   → DBSCAN: {n_clusters_dbscan} clusters, {n_outliers} outliers")
        
        if n_outliers > 0 and n_outliers < len(X) * 0.1:  # Menos del 10% outliers
            restriccion = RestriccionAprendida(
                id=generar_id(),
                tipo='patron_outlier',
                descripcion=f"Detectados {n_outliers} casos atípicos que no siguen patrones habituales",
                entidades={
                    'n_outliers': int(n_outliers),
                    'porcentaje': n_outliers / len(X) * 100
                },
                soporte=n_outliers / len(X),
                confianza=0.9,
                penalizacion_incumplimiento=0.1,
                es_hard_constraint=False
            )
            restricciones.append(restriccion)
        
        return restricciones
    
    def _find_optimal_k(self, X: np.ndarray, max_k: int = 10) -> int:
        """Encuentra el K óptimo usando el método del codo"""
        inertias = []
        K_range = range(2, min(max_k + 1, len(X) // 10))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Encontrar el "codo" (punto de máxima curvatura)
        if len(inertias) < 3:
            return 3
        
        # Calcular segunda derivada para encontrar el codo
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        
        elbow_idx = np.argmax(diffs2) + 2  # +2 porque empezamos en k=2
        
        return max(3, min(elbow_idx + 2, max_k))
    
    def _caracterizar_cluster(self, cluster_data: pd.DataFrame, 
                             all_data: pd.DataFrame) -> List[str]:
        """Identifica características distintivas de un cluster"""
        caracteristicas = []
        
        for col in cluster_data.columns:
            if col == 'cluster':
                continue
            
            try:
                if cluster_data[col].dtype in ['int64', 'float64']:
                    # Variable numérica: comparar medias
                    cluster_mean = cluster_data[col].mean()
                    global_mean = all_data[col].mean()
                    global_std = all_data[col].std()
                    
                    if global_std > 0:
                        z_score = (cluster_mean - global_mean) / global_std
                        if abs(z_score) > 1.5:
                            direccion = "alto" if z_score > 0 else "bajo"
                            caracteristicas.append(f"{col} {direccion} ({cluster_mean:.1f})")
                else:
                    # Variable categórica: encontrar moda dominante
                    moda = cluster_data[col].mode()
                    if len(moda) > 0:
                        moda_val = moda.iloc[0]
                        prop_cluster = (cluster_data[col] == moda_val).mean()
                        prop_global = (all_data[col] == moda_val).mean()
                        
                        if prop_cluster > prop_global * 1.5 and prop_cluster > 0.5:
                            caracteristicas.append(f"{col}={moda_val} ({prop_cluster*100:.0f}%)")
            except:
                continue
        
        return caracteristicas[:5]  # Top 5 características
    
    # =========================================================================
    # 3. ÁRBOLES DE DECISIÓN
    # =========================================================================
    
    def _decision_tree_rules(self, df: pd.DataFrame) -> List[RestriccionAprendida]:
        """
        Usa árboles de decisión para descubrir reglas interpretables.
        Ejemplo: SI complejidad > 3 Y especialidad = digestiva → quirófano 1 o 2
        """
        restricciones = []
        
        # Objetivo 1: Predecir quirófano asignado
        if 'quirofano_id' in df.columns:
            reglas = self._extract_decision_rules(
                df, 
                target='quirofano_id',
                features=['especialidad', 'complejidad', 'dia_semana', 'tipo_intervencion']
            )
            restricciones.extend(reglas)
        
        # Objetivo 2: Predecir overtime
        if 'tiene_overtime' in df.columns:
            reglas = self._extract_decision_rules(
                df,
                target='tiene_overtime', 
                features=['duracion_programada', 'hora_inicio', 'complejidad', 'especialidad']
            )
            restricciones.extend(reglas)
        
        # Objetivo 3: Predecir complicaciones
        if 'complicacion' in df.columns:
            reglas = self._extract_decision_rules(
                df,
                target='complicacion',
                features=['paciente_asa', 'paciente_edad', 'complejidad', 'duracion_real']
            )
            restricciones.extend(reglas)
        
        return restricciones
    
    def _extract_decision_rules(self, df: pd.DataFrame, target: str, 
                                features: List[str]) -> List[RestriccionAprendida]:
        """Extrae reglas interpretables de un árbol de decisión"""
        restricciones = []
        
        features_disponibles = [f for f in features if f in df.columns]
        if len(features_disponibles) < 2 or target not in df.columns:
            return restricciones
        
        # Preparar datos
        df_clean = df[features_disponibles + [target]].dropna()
        if len(df_clean) < 50:
            return restricciones
        
        X = df_clean[features_disponibles].copy()
        y = df_clean[target]
        
        # Codificar variables categóricas
        for col in X.columns:
            if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self._encoders[col] = le
        
        # Entrenar árbol de decisión (poco profundo para interpretabilidad)
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
        
        try:
            tree.fit(X, y)
        except Exception as e:
            print(f"   ⚠️ Error entrenando árbol para {target}: {e}")
            return restricciones
        
        # Extraer reglas del árbol
        reglas_texto = self._tree_to_rules(tree, features_disponibles, X, y)
        
        print(f"   → Árbol para '{target}': {len(reglas_texto)} reglas extraídas")
        
        for regla in reglas_texto[:5]:  # Top 5 reglas
            restriccion = RestriccionAprendida(
                id=generar_id(),
                tipo='regla_decision_tree',
                descripcion=regla['descripcion'],
                entidades={
                    'target': target,
                    'condiciones': regla['condiciones'],
                    'prediccion': regla['prediccion'],
                    'n_samples': regla['samples']
                },
                soporte=regla['samples'] / len(df),
                confianza=regla['confianza'],
                penalizacion_incumplimiento=0.3,
                es_hard_constraint=False
            )
            restricciones.append(restriccion)
        
        return restricciones
    
    def _tree_to_rules(self, tree, feature_names: List[str], 
                       X: pd.DataFrame, y: pd.Series) -> List[Dict]:
        """Convierte un árbol de decisión a reglas legibles"""
        reglas = []
        
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        
        def recurse(node, condiciones_actuales):
            if tree_.feature[node] != -2:  # Nodo interno
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # Rama izquierda (<=)
                recurse(
                    tree_.children_left[node],
                    condiciones_actuales + [(name, '<=', threshold)]
                )
                # Rama derecha (>)
                recurse(
                    tree_.children_right[node],
                    condiciones_actuales + [(name, '>', threshold)]
                )
            else:  # Hoja
                samples = tree_.n_node_samples[node]
                if samples >= 20:  # Mínimo de muestras
                    values = tree_.value[node][0]
                    prediccion_idx = np.argmax(values)
                    confianza = values[prediccion_idx] / sum(values)
                    
                    if confianza >= 0.6:  # Mínimo 60% confianza
                        # Decodificar predicción si es posible
                        try:
                            clases = tree.classes_
                            prediccion = clases[prediccion_idx]
                        except:
                            prediccion = prediccion_idx
                        
                        # Formatear condiciones
                        cond_texto = ' Y '.join([
                            f"{c[0]} {c[1]} {c[2]:.1f}" for c in condiciones_actuales
                        ])
                        
                        reglas.append({
                            'descripcion': f"SI {cond_texto} → {prediccion} (conf: {confianza:.0%})",
                            'condiciones': condiciones_actuales,
                            'prediccion': str(prediccion),
                            'samples': samples,
                            'confianza': confianza
                        })
        
        recurse(0, [])
        
        # Ordenar por confianza
        reglas.sort(key=lambda x: x['confianza'], reverse=True)
        
        return reglas
    
    # =========================================================================
    # 4. DETECCIÓN DE ANOMALÍAS
    # =========================================================================
    
    def _anomaly_detection(self, df: pd.DataFrame) -> List[RestriccionAprendida]:
        """
        Detecta anomalías usando Isolation Forest.
        Identifica patrones inusuales que podrían indicar restricciones implícitas.
        """
        restricciones = []
        
        features = ['hora_inicio', 'duracion_real', 'quirofano_id', 'complejidad']
        features_disponibles = [f for f in features if f in df.columns]
        
        if len(features_disponibles) < 2:
            return restricciones
        
        X = df[features_disponibles].dropna()
        if len(X) < 100:
            return restricciones
        
        # Entrenar Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomalias = iso_forest.fit_predict(X)
        
        # Analizar anomalías
        df_anomalias = X[anomalias == -1]
        df_normales = X[anomalias == 1]
        
        n_anomalias = len(df_anomalias)
        print(f"   → {n_anomalias} anomalías detectadas ({n_anomalias/len(X)*100:.1f}%)")
        
        if n_anomalias > 5:
            # Caracterizar las anomalías
            for col in features_disponibles:
                media_anomalia = df_anomalias[col].mean()
                media_normal = df_normales[col].mean()
                std_normal = df_normales[col].std()
                
                if std_normal > 0:
                    diferencia = abs(media_anomalia - media_normal) / std_normal
                    
                    if diferencia > 1.5:
                        direccion = "superiores" if media_anomalia > media_normal else "inferiores"
                        
                        restriccion = RestriccionAprendida(
                            id=generar_id(),
                            tipo='patron_anomalia',
                            descripcion=f"Casos atípicos tienen {col} {direccion} a lo normal ({media_anomalia:.1f} vs {media_normal:.1f})",
                            entidades={
                                'variable': col,
                                'media_anomalia': media_anomalia,
                                'media_normal': media_normal,
                                'desviacion_std': diferencia
                            },
                            soporte=n_anomalias / len(X),
                            confianza=0.85,
                            penalizacion_incumplimiento=0.4,
                            es_hard_constraint=False
                        )
                        restricciones.append(restriccion)
        
        return restricciones
    
    # =========================================================================
    # 5. CORRELACIONES MULTI-VARIABLE
    # =========================================================================
    
    def _multivariate_correlations(self, df: pd.DataFrame) -> List[RestriccionAprendida]:
        """
        Detecta correlaciones complejas entre múltiples variables.
        Usa PCA y análisis de correlación parcial.
        """
        restricciones = []
        
        # Variables numéricas
        vars_numericas = ['hora_inicio', 'duracion_real', 'duracion_programada', 
                         'complejidad', 'paciente_edad', 'paciente_asa', 'overtime']
        vars_disponibles = [v for v in vars_numericas if v in df.columns]
        
        if len(vars_disponibles) < 3:
            return restricciones
        
        X = df[vars_disponibles].dropna()
        if len(X) < 100:
            return restricciones
        
        # Matriz de correlación
        corr_matrix = X.corr()
        
        # Encontrar correlaciones fuertes (>0.5 o <-0.5)
        for i, var1 in enumerate(vars_disponibles):
            for j, var2 in enumerate(vars_disponibles):
                if i < j:  # Solo triangular superior
                    corr = corr_matrix.loc[var1, var2]
                    if abs(corr) > 0.5:
                        tipo_corr = "positiva" if corr > 0 else "negativa"
                        
                        restriccion = RestriccionAprendida(
                            id=generar_id(),
                            tipo='correlacion_bivariada',
                            descripcion=f"Correlación {tipo_corr} entre {var1} y {var2} (r={corr:.2f})",
                            entidades={
                                'variable_1': var1,
                                'variable_2': var2,
                                'correlacion': corr
                            },
                            soporte=len(X) / len(df),
                            confianza=abs(corr),
                            penalizacion_incumplimiento=0.2,
                            es_hard_constraint=False
                        )
                        restricciones.append(restriccion)
        
        # PCA para encontrar combinaciones de variables importantes
        if len(vars_disponibles) >= 4:
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=min(3, len(vars_disponibles)))
            pca.fit(X_scaled)
            
            for i, (comp, var_exp) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
                if var_exp > 0.2:  # Componente explica >20% varianza
                    # Encontrar variables dominantes en este componente
                    vars_importantes = [(vars_disponibles[j], abs(comp[j])) 
                                       for j in range(len(comp)) if abs(comp[j]) > 0.4]
                    vars_importantes.sort(key=lambda x: x[1], reverse=True)
                    
                    if vars_importantes:
                        vars_texto = ', '.join([v[0] for v in vars_importantes[:3]])
                        
                        restriccion = RestriccionAprendida(
                            id=generar_id(),
                            tipo='componente_principal',
                            descripcion=f"Patrón multivariable PC{i+1}: {vars_texto} (explica {var_exp*100:.0f}% varianza)",
                            entidades={
                                'componente': i + 1,
                                'varianza_explicada': var_exp,
                                'variables_principales': [v[0] for v in vars_importantes],
                                'pesos': [v[1] for v in vars_importantes]
                            },
                            soporte=var_exp,
                            confianza=0.9,
                            penalizacion_incumplimiento=0.1,
                            es_hard_constraint=False
                        )
                        restricciones.append(restriccion)
        
        print(f"   → {len(restricciones)} patrones de correlación descubiertos")
        
        return restricciones
    
    # =========================================================================
    # 6. PATRONES TEMPORALES AVANZADOS
    # =========================================================================
    
    def _temporal_pattern_mining(self, df: pd.DataFrame) -> List[RestriccionAprendida]:
        """
        Descubre patrones temporales complejos.
        """
        restricciones = []
        
        if 'fecha' not in df.columns or 'hora_inicio' not in df.columns:
            return restricciones
        
        # Patrón: Distribución de cirugías por hora según día de semana
        if 'dia_semana' in df.columns:
            for dia in range(5):  # Lunes a viernes
                df_dia = df[df['dia_semana'] == dia]
                if len(df_dia) < 20:
                    continue
                
                hora_media = df_dia['hora_inicio'].mean()
                hora_global = df['hora_inicio'].mean()
                
                diferencia = abs(hora_media - hora_global)
                if diferencia > 30:  # Más de 30 min de diferencia
                    dias_nombre = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
                    direccion = "más temprano" if hora_media < hora_global else "más tarde"
                    
                    restriccion = RestriccionAprendida(
                        id=generar_id(),
                        tipo='patron_temporal_dia',
                        descripcion=f"{dias_nombre[dia]}: cirugías empiezan {direccion} ({self._min_to_time(hora_media)} vs {self._min_to_time(hora_global)} promedio)",
                        entidades={
                            'dia_semana': dia,
                            'hora_media_dia': hora_media,
                            'hora_media_global': hora_global
                        },
                        soporte=len(df_dia) / len(df),
                        confianza=0.8,
                        penalizacion_incumplimiento=0.1,
                        es_hard_constraint=False
                    )
                    restricciones.append(restriccion)
        
        # Patrón: Tipos de cirugía por franja horaria
        if 'franja_horaria' in df.columns and 'tipo_intervencion' in df.columns:
            crosstab = pd.crosstab(df['tipo_intervencion'], df['franja_horaria'], normalize='index')
            
            for interv in crosstab.index:
                for franja in crosstab.columns:
                    prop = crosstab.loc[interv, franja]
                    prop_global = (df['franja_horaria'] == franja).mean()
                    
                    if prop > prop_global * 2 and prop > 0.4:  # Doble que el promedio y >40%
                        restriccion = RestriccionAprendida(
                            id=generar_id(),
                            tipo='patron_intervencion_horario',
                            descripcion=f"{interv} se programa principalmente en {franja} ({prop*100:.0f}% vs {prop_global*100:.0f}% global)",
                            entidades={
                                'intervencion': interv,
                                'franja': str(franja),
                                'proporcion': prop
                            },
                            soporte=(df['tipo_intervencion'] == interv).sum() / len(df),
                            confianza=prop,
                            lift=prop / prop_global if prop_global > 0 else 1,
                            penalizacion_incumplimiento=0.3,
                            es_hard_constraint=False
                        )
                        restricciones.append(restriccion)
        
        print(f"   → {len(restricciones)} patrones temporales descubiertos")
        
        return restricciones
    
    def _min_to_time(self, minutos: float) -> str:
        """Convierte minutos desde medianoche a formato HH:MM"""
        h, m = divmod(int(minutos), 60)
        return f"{h:02d}:{m:02d}"
    
    # =========================================================================
    # 7. SEQUENTIAL PATTERN MINING
    # =========================================================================
    
    def _sequential_pattern_mining(self, df: pd.DataFrame) -> List[RestriccionAprendida]:
        """
        Descubre patrones de secuencia en las cirugías.
        Ejemplo: Después de cirugía tipo A, suele ir tipo B en el mismo quirófano.
        """
        restricciones = []
        
        if 'fecha' not in df.columns or 'quirofano_id' not in df.columns:
            return restricciones
        
        if 'hora_inicio' not in df.columns or 'tipo_intervencion' not in df.columns:
            return restricciones
        
        # Ordenar por fecha y hora
        df_sorted = df.sort_values(['fecha', 'quirofano_id', 'hora_inicio'])
        
        # Encontrar pares consecutivos en el mismo quirófano y día
        secuencias = defaultdict(int)
        total_secuencias = 0
        
        for (fecha, quirofano), grupo in df_sorted.groupby(['fecha', 'quirofano_id']):
            if len(grupo) < 2:
                continue
            
            tipos = grupo['tipo_intervencion'].tolist()
            for i in range(len(tipos) - 1):
                secuencia = (tipos[i], tipos[i+1])
                secuencias[secuencia] += 1
                total_secuencias += 1
        
        if total_secuencias < 50:
            return restricciones
        
        # Encontrar secuencias frecuentes
        for (tipo1, tipo2), count in secuencias.items():
            soporte = count / total_secuencias
            
            # Calcular confianza: P(tipo2 | tipo1)
            n_tipo1 = (df['tipo_intervencion'] == tipo1).sum()
            confianza = count / n_tipo1 if n_tipo1 > 0 else 0
            
            if soporte > 0.05 and confianza > 0.3:  # >5% soporte, >30% confianza
                restriccion = RestriccionAprendida(
                    id=generar_id(),
                    tipo='patron_secuencia',
                    descripcion=f"Después de {tipo1} suele programarse {tipo2} ({confianza*100:.0f}% de veces)",
                    entidades={
                        'cirugia_previa': tipo1,
                        'cirugia_siguiente': tipo2,
                        'frecuencia': count
                    },
                    soporte=soporte,
                    confianza=confianza,
                    penalizacion_incumplimiento=0.2,
                    es_hard_constraint=False
                )
                restricciones.append(restriccion)
        
        print(f"   → {len(restricciones)} patrones de secuencia descubiertos")
        
        return restricciones
    
    # =========================================================================
    # MÉTODOS DE UTILIDAD
    # =========================================================================
    
    def generar_resumen(self) -> str:
        """Genera un resumen detallado de las restricciones aprendidas."""
        if not self.restricciones_aprendidas:
            return "No se han aprendido restricciones todavía."
        
        lineas = [
            "=" * 70,
            "RESUMEN DE RESTRICCIONES APRENDIDAS (ML AVANZADO)",
            "=" * 70,
            f"Total de restricciones: {len(self.restricciones_aprendidas)}",
            ""
        ]
        
        # Agrupar por tipo
        por_tipo = defaultdict(list)
        for r in self.restricciones_aprendidas:
            por_tipo[r.tipo].append(r)
        
        for tipo, restricciones in sorted(por_tipo.items()):
            lineas.append(f"\n[{tipo.upper()}] ({len(restricciones)} restricciones)")
            
            # Ordenar por confianza
            restricciones_ord = sorted(restricciones, key=lambda x: x.confianza, reverse=True)
            
            for r in restricciones_ord[:3]:  # Top 3 por tipo
                lineas.append(f"  • {r.descripcion}")
                lineas.append(f"    (soporte: {r.soporte:.1%}, confianza: {r.confianza:.1%}, lift: {r.lift:.2f})")
            
            if len(restricciones) > 3:
                lineas.append(f"  ... y {len(restricciones) - 3} más")
        
        return "\n".join(lineas)
    
    def exportar_restricciones(self) -> pd.DataFrame:
        """Exporta las restricciones a un DataFrame."""
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
    """Prueba del módulo de aprendizaje avanzado"""
    from synthetic_data import GeneradorDatosSinteticos
    
    print("Generando datos sintéticos...")
    generador = GeneradorDatosSinteticos(seed=42)
    cirujanos, _, historico = generador.generar_dataset_completo(
        n_solicitudes_espera=100,
        dias_historico=365,
        cirugias_dia_historico=25
    )
    
    print(f"\nDataset: {len(historico)} registros históricos\n")
    
    aprendizaje = AprendizajeRestriccionesAvanzado()
    restricciones = aprendizaje.analizar_historico(historico)
    
    print("\n" + aprendizaje.generar_resumen())
    
    return aprendizaje


if __name__ == "__main__":
    main()
