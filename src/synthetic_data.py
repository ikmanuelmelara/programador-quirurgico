"""
Generador de Datos SintÃ©ticos para el Programador QuirÃºrgico
============================================================
Genera datos realistas de pacientes, cirugÃ­as, cirujanos y
registros histÃ³ricos para entrenamiento y pruebas.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict

from config import (
    Especialidad, PrioridadCatSalut, CATALOGO_INTERVENCIONES,
    TipoIntervencion, TIEMPOS_MAXIMOS_ESPERA, Quirofano, QUIROFANOS_DEFAULT
)
from models import (
    Paciente, Cirujano, SolicitudCirugia, CirugiaProgramada,
    ProgramaDiario, ClaseASA, Comorbilidad, EstadoCirugia,
    generar_id_paciente, generar_id_solicitud, generar_id_cirugia
)


NOMBRES_MASCULINOS = [
    "Marc", "Jordi", "Joan", "Josep", "David", "Carles", "Pau", "Albert",
    "Xavier", "Miquel", "Pere", "Ramon", "Francesc", "Antoni", "Enric",
    "Sergi", "Oriol", "Roger", "Arnau", "Gerard", "MartÃ­", "AdriÃ "
]

NOMBRES_FEMENINOS = [
    "Maria", "Anna", "Montserrat", "Carme", "Rosa", "Marta", "Laura",
    "NÃºria", "Cristina", "Elena", "SÃ­lvia", "Laia", "Alba", "Sara",
    "Paula", "Andrea", "JÃºlia", "Emma", "Claudia", "Marina"
]

APELLIDOS = [
    "GarcÃ­a", "MartÃ­nez", "LÃ³pez", "SÃ¡nchez", "GonzÃ¡lez", "RodrÃ­guez",
    "Serra", "Puig", "Ferrer", "Roca", "Soler", "Vila", "Font", "Mas",
    "Sala", "Vidal", "Costa", "Pons", "Torres", "Ribas", "Bosch"
]

DIAGNOSTICOS_POR_INTERVENCION = {
    "COL_LAP": ["Colelitiasis sintomÃ¡tica", "Colecistitis crÃ³nica"],
    "HERN_ING": ["Hernia inguinal indirecta", "Hernia inguinal directa"],
    "COLECT_DER": ["Adenocarcinoma de colon derecho", "Tumor de ciego"],
    "COLECT_IZQ": ["Adenocarcinoma de colon izquierdo", "Tumor de sigma"],
    "RECT_ANT": ["Adenocarcinoma de recto", "Tumor rectal"],
    "MAST_SIMPLE": ["Carcinoma ductal in situ", "Carcinoma lobulillar"],
    "TUMORECT": ["Carcinoma mama estadio I", "Tumor mama pequeÃ±o"],
    "TIROID_TOTAL": ["Carcinoma papilar tiroides", "Bocio multinodular"],
    "PROST_RAD": ["Adenocarcinoma prÃ³stata", "Carcinoma prostÃ¡tico"],
    "HIST_LAP": ["Ãštero miomatoso", "Adenomiosis"],
    "BYPASS_G": ["Obesidad mÃ³rbida", "Obesidad con comorbilidades"],
    "SLEEVE": ["Obesidad grado III", "Obesidad con DM2"],
}


class GeneradorDatosSinteticos:
    """Generador de datos sintÃ©ticos realistas para programaciÃ³n quirÃºrgica."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self._configurar_distribuciones()
    
    def _configurar_distribuciones(self):
        """Configura las distribuciones de probabilidad"""
        self.dist_edad = {
            'general': (55, 18),
            'oncologico': (62, 12),
            'bariatrico': (42, 10),
        }
        
        self.prob_asa_por_edad = {
            'joven': [0.6, 0.3, 0.08, 0.02, 0.0],
            'medio': [0.3, 0.4, 0.25, 0.04, 0.01],
            'mayor': [0.1, 0.3, 0.4, 0.15, 0.05],
            'anciano': [0.05, 0.2, 0.4, 0.25, 0.1],
        }
        
        self.frecuencia_intervenciones = {
            "COL_LAP": 15, "HERN_ING": 12, "HERN_UMB": 5, "HERN_INC": 4,
            "TUMORECT": 8, "BSGC": 6, "TIROID_PARC": 5, "TIROID_TOTAL": 4,
            "RTU_PROST": 8, "RTU_VES": 6, "HIST_LAP": 7, "VARIC": 8,
            "COLECT_DER": 3, "COLECT_IZQ": 3, "MAST_SIMPLE": 3,
            "HEMORR": 5, "FISTULA_AN": 4, "PILONIDAL": 4, "OOFOR": 4,
            "SLEEVE": 3, "BYPASS_G": 2, "RECT_ANT": 2, "PROST_RAD": 3,
            "NEFR_PARC": 2, "APEND_LAP": 4, "MAST_RAD": 2, "PARATIR": 2,
            "HEPAT_SEG": 1, "GASTRECT": 1, "NEFR_RAD": 1, "ENDART": 2,
            "EXCISION": 6, "HIST_ABD": 3, "HIST_VAG": 2, "MIOM": 2,
        }
        
        total = sum(self.frecuencia_intervenciones.values())
        self.prob_intervenciones = {
            k: v/total for k, v in self.frecuencia_intervenciones.items()
        }
        
        self.dist_prioridad = {
            'oncologico': {
                PrioridadCatSalut.ONCOLOGICO_PRIORITARIO: 0.7,
                PrioridadCatSalut.ONCOLOGICO_ESTANDAR: 0.3,
            },
            'standard': {
                PrioridadCatSalut.REFERENCIA_P1: 0.2,
                PrioridadCatSalut.REFERENCIA_P2: 0.5,
                PrioridadCatSalut.REFERENCIA_P3: 0.3,
            }
        }
    
    def generar_paciente(self, tipo_cirugia: str = None) -> Paciente:
        """Genera un paciente sintÃ©tico con caracterÃ­sticas realistas."""
        # Determinar sexo
        if tipo_cirugia:
            interv = CATALOGO_INTERVENCIONES.get(tipo_cirugia)
            if interv:
                esp = interv.especialidad
                if esp == Especialidad.GINECOLOGIA:
                    sexo = 'F'
                elif esp == Especialidad.UROLOGIA and tipo_cirugia in ['PROST_RAD', 'RTU_PROST']:
                    sexo = 'M'
                elif esp == Especialidad.CIRUGIA_MAMA:
                    sexo = 'F' if random.random() < 0.99 else 'M'
                else:
                    sexo = random.choice(['M', 'F'])
            else:
                sexo = random.choice(['M', 'F'])
        else:
            sexo = random.choice(['M', 'F'])
        
        nombre = random.choice(NOMBRES_MASCULINOS if sexo == 'M' else NOMBRES_FEMENINOS)
        nombre_completo = f"{nombre} {random.choice(APELLIDOS)} {random.choice(APELLIDOS)}"
        
        # Edad
        if tipo_cirugia in ['BYPASS_G', 'SLEEVE', 'BANDA_G']:
            edad_media, edad_std = self.dist_edad['bariatrico']
        elif tipo_cirugia and CATALOGO_INTERVENCIONES.get(tipo_cirugia):
            interv = CATALOGO_INTERVENCIONES[tipo_cirugia]
            if interv.prioridad_tipica in [PrioridadCatSalut.ONCOLOGICO_PRIORITARIO,
                                           PrioridadCatSalut.ONCOLOGICO_ESTANDAR]:
                edad_media, edad_std = self.dist_edad['oncologico']
            else:
                edad_media, edad_std = self.dist_edad['general']
        else:
            edad_media, edad_std = self.dist_edad['general']
        
        edad = int(np.clip(np.random.normal(edad_media, edad_std), 18, 95))
        fecha_nacimiento = date.today() - timedelta(days=edad*365 + random.randint(0, 364))
        
        # ASA segÃºn edad
        if edad < 40:
            grupo = 'joven'
        elif edad < 65:
            grupo = 'medio'
        elif edad < 75:
            grupo = 'mayor'
        else:
            grupo = 'anciano'
        
        asa_idx = np.random.choice(5, p=self.prob_asa_por_edad[grupo])
        clase_asa = list(ClaseASA)[asa_idx]
        
        # Comorbilidades
        comorbilidades = []
        prob_base = 0.1 + (asa_idx * 0.15)
        if random.random() < prob_base:
            comorbilidades.append(Comorbilidad.HIPERTENSION)
        if random.random() < prob_base * 0.7:
            comorbilidades.append(Comorbilidad.DIABETES)
        if random.random() < prob_base * 0.5:
            comorbilidades.append(Comorbilidad.CARDIOPATIA)
        if tipo_cirugia in ['BYPASS_G', 'SLEEVE']:
            comorbilidades.append(Comorbilidad.OBESIDAD_MORBIDA)
        
        return Paciente(
            id=generar_id_paciente(),
            nombre=nombre_completo,
            fecha_nacimiento=fecha_nacimiento,
            sexo=sexo,
            numero_historia=f"HC{random.randint(100000, 999999)}",
            clase_asa=clase_asa,
            comorbilidades=comorbilidades,
            telefono=f"6{random.randint(10000000, 99999999)}",
            email=f"{nombre.lower()}.{random.choice(APELLIDOS).lower()}@email.com"
        )
    
    def generar_cirujano(self, especialidad: Especialidad, id_num: int) -> Cirujano:
        """Genera un cirujano sintÃ©tico."""
        sexo = random.choice(['M', 'F'])
        nombre = random.choice(NOMBRES_MASCULINOS if sexo == 'M' else NOMBRES_FEMENINOS)
        nombre_completo = f"Dr. {nombre} {random.choice(APELLIDOS)}"
        
        nivel = int(np.clip(np.random.normal(3.5, 1), 1, 5))
        
        intervenciones = [
            codigo for codigo, interv in CATALOGO_INTERVENCIONES.items()
            if interv.especialidad == especialidad
        ]
        
        quirofanos_pref = [
            q.id for q in QUIROFANOS_DEFAULT
            if especialidad in q.especialidades_permitidas
        ]
        
        return Cirujano(
            id=f"CIR-{id_num:03d}",
            nombre=nombre_completo,
            especialidad_principal=especialidad,
            intervenciones_habilitadas=intervenciones,
            nivel_experiencia=nivel,
            quirofanos_preferidos=quirofanos_pref
        )
    
    def generar_equipo_cirujanos(self) -> List[Cirujano]:
        """Genera un equipo completo de cirujanos."""
        cirujanos = []
        id_num = 1
        
        cirujanos_por_esp = {
            Especialidad.CIRUGIA_GENERAL: 4,
            Especialidad.CIRUGIA_DIGESTIVA: 4,
            Especialidad.CIRUGIA_COLORRECTAL: 3,
            Especialidad.CIRUGIA_HEPATOBILIAR: 2,
            Especialidad.CIRUGIA_MAMA: 3,
            Especialidad.CIRUGIA_ENDOCRINA: 2,
            Especialidad.CIRUGIA_BARIATRICA: 2,
            Especialidad.UROLOGIA: 4,
            Especialidad.GINECOLOGIA: 4,
            Especialidad.CIRUGIA_VASCULAR: 3,
            Especialidad.CIRUGIA_PLASTICA: 2,
        }
        
        for esp, num in cirujanos_por_esp.items():
            for _ in range(num):
                cirujanos.append(self.generar_cirujano(esp, id_num))
                id_num += 1
        
        return cirujanos
    
    def generar_solicitud_cirugia(
        self, 
        cirujanos: List[Cirujano],
        fecha_indicacion: date = None,
        tipo_intervencion: str = None
    ) -> SolicitudCirugia:
        """Genera una solicitud de cirugÃ­a sintÃ©tica."""
        if tipo_intervencion is None:
            codigos = list(self.prob_intervenciones.keys())
            probs = list(self.prob_intervenciones.values())
            tipo_intervencion = np.random.choice(codigos, p=probs)
        
        if tipo_intervencion not in CATALOGO_INTERVENCIONES:
            tipo_intervencion = random.choice(list(CATALOGO_INTERVENCIONES.keys()))
        
        tipo_info = CATALOGO_INTERVENCIONES[tipo_intervencion]
        paciente = self.generar_paciente(tipo_intervencion)
        
        if fecha_indicacion is None:
            dias_atras = int(np.random.exponential(90))
            dias_atras = min(dias_atras, 400)
            fecha_indicacion = date.today() - timedelta(days=dias_atras)
        
        # Prioridad
        if tipo_info.prioridad_tipica in [PrioridadCatSalut.ONCOLOGICO_PRIORITARIO,
                                          PrioridadCatSalut.ONCOLOGICO_ESTANDAR]:
            prioridad = np.random.choice(
                list(self.dist_prioridad['oncologico'].keys()),
                p=list(self.dist_prioridad['oncologico'].values())
            )
        else:
            prioridad = np.random.choice(
                list(self.dist_prioridad['standard'].keys()),
                p=list(self.dist_prioridad['standard'].values())
            )
        
        cirujanos_validos = [
            c for c in cirujanos 
            if tipo_info.especialidad == c.especialidad_principal
        ]
        cirujano = random.choice(cirujanos_validos) if cirujanos_validos else None
        
        diagnosticos = DIAGNOSTICOS_POR_INTERVENCION.get(
            tipo_intervencion, [f"PatologÃ­a para {tipo_info.nombre}"]
        )
        
        dias_desde = (date.today() - fecha_indicacion).days
        preop = dias_desde > 7 and random.random() < 0.85
        
        solicitud = SolicitudCirugia(
            id=generar_id_solicitud(),
            paciente=paciente,
            tipo_intervencion=tipo_info,
            fecha_indicacion=fecha_indicacion,
            prioridad=prioridad,
            cirujano_solicitante=cirujano,
            cirujano_asignado=cirujano,
            diagnostico_principal=random.choice(diagnosticos),
            preoperatorio_completado=preop,
            consentimiento_firmado=preop and random.random() < 0.95
        )
        
        solicitud.calcular_score_clinico()
        return solicitud
    
    def generar_lista_espera(
        self, cirujanos: List[Cirujano], n_solicitudes: int = 200
    ) -> List[SolicitudCirugia]:
        """Genera una lista de espera completa."""
        solicitudes = [
            self.generar_solicitud_cirugia(cirujanos) 
            for _ in range(n_solicitudes)
        ]
        solicitudes.sort(key=lambda x: x.score_clinico, reverse=True)
        return solicitudes
    
    def generar_historico_cirugias(
        self,
        cirujanos: List[Cirujano],
        dias: int = 365,
        cirugias_por_dia: int = 25
    ) -> pd.DataFrame:
        """Genera un histÃ³rico de cirugÃ­as para entrenamiento."""
        registros = []
        fecha_inicio = date.today() - timedelta(days=dias)
        
        for dia in range(dias):
            fecha = fecha_inicio + timedelta(days=dia)
            if fecha.weekday() >= 5:
                continue
            
            n_cirugias = max(0, int(np.random.normal(cirugias_por_dia, 5)))
            
            for _ in range(n_cirugias):
                solicitud = self.generar_solicitud_cirugia(
                    cirujanos, 
                    fecha_indicacion=fecha - timedelta(days=random.randint(7, 180))
                )
                
                quirofanos_validos = [
                    q.id for q in QUIROFANOS_DEFAULT
                    if solicitud.tipo_intervencion.especialidad in q.especialidades_permitidas
                ]
                quirofano_id = random.choice(quirofanos_validos) if quirofanos_validos else 1
                
                hora_inicio = 8 * 60 + random.randint(0, 300)  # 8:00-13:00
                
                duracion_prog = solicitud.tipo_intervencion.duracion_media_min
                factor = np.random.lognormal(0, 0.2)
                duracion_real = int(duracion_prog * factor)
                duracion_real = max(15, min(duracion_real, duracion_prog * 2))
                
                prob_compl = 0.05 + (solicitud.paciente.clase_asa.value - 1) * 0.02
                complicacion = random.random() < prob_compl
                
                registros.append({
                    'fecha': fecha,
                    'dia_semana': fecha.weekday(),
                    'quirofano_id': quirofano_id,
                    'hora_inicio': hora_inicio,
                    'hora_fin': hora_inicio + duracion_real,
                    'duracion_programada': duracion_prog,
                    'duracion_real': duracion_real,
                    'tipo_intervencion': solicitud.tipo_intervencion.codigo,
                    'especialidad': solicitud.tipo_intervencion.especialidad.name,
                    'cirujano_id': solicitud.cirujano_asignado.id if solicitud.cirujano_asignado else None,
                    'paciente_edad': solicitud.paciente.edad,
                    'paciente_sexo': solicitud.paciente.sexo,
                    'paciente_asa': solicitud.paciente.clase_asa.value,
                    'prioridad': solicitud.prioridad.name,
                    'dias_espera': (fecha - solicitud.fecha_indicacion).days,
                    'complejidad': solicitud.tipo_intervencion.complejidad,
                    'requiere_uci': solicitud.tipo_intervencion.requiere_uci,
                    'ingreso_uci': complicacion and solicitud.tipo_intervencion.requiere_uci,
                    'complicacion': complicacion,
                    'overtime': max(0, hora_inicio + duracion_real - 15*60),
                })
        
        return pd.DataFrame(registros)
    
    def generar_dataset_completo(
        self,
        n_solicitudes_espera: int = 250,
        dias_historico: int = 365,
        cirugias_dia_historico: int = 25
    ) -> Tuple[List[Cirujano], List[SolicitudCirugia], pd.DataFrame]:
        """Genera un dataset completo para el sistema."""
        print("Generando equipo de cirujanos...")
        cirujanos = self.generar_equipo_cirujanos()
        print(f"  -> {len(cirujanos)} cirujanos generados")
        
        print("Generando lista de espera...")
        lista_espera = self.generar_lista_espera(cirujanos, n_solicitudes_espera)
        print(f"  -> {len(lista_espera)} solicitudes generadas")
        
        fuera_plazo = sum(1 for s in lista_espera if s.esta_fuera_plazo)
        oncologicas = sum(1 for s in lista_espera 
                        if s.prioridad in [PrioridadCatSalut.ONCOLOGICO_PRIORITARIO,
                                          PrioridadCatSalut.ONCOLOGICO_ESTANDAR])
        print(f"     {fuera_plazo} fuera de plazo ({fuera_plazo/len(lista_espera)*100:.1f}%)")
        print(f"     {oncologicas} oncolÃ³gicas ({oncologicas/len(lista_espera)*100:.1f}%)")
        
        print("Generando histÃ³rico de cirugÃ­as...")
        historico = self.generar_historico_cirugias(
            cirujanos, dias_historico, cirugias_dia_historico
        )
        print(f"  -> {len(historico)} registros histÃ³ricos generados")
        
        return cirujanos, lista_espera, historico


if __name__ == "__main__":
    generador = GeneradorDatosSinteticos(seed=42)
    cirujanos, lista_espera, historico = generador.generar_dataset_completo()
    print("\nDataset generado exitosamente.")
