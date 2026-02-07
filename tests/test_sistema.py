"""
Tests para el Programador Quirúrgico Inteligente
================================================

Ejecutar con: pytest tests/ -v
"""

import pytest
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestConfig:
    """Tests para config.py"""
    
    def test_prioridades_catsalut_existen(self):
        """Verificar que las prioridades CatSalut están definidas"""
        from config import PrioridadCatSalut
        
        assert hasattr(PrioridadCatSalut, 'ONCOLOGICO_PRIORITARIO')
        assert hasattr(PrioridadCatSalut, 'ONCOLOGICO_ESTANDAR')
        assert hasattr(PrioridadCatSalut, 'REFERENCIA_P1')
        assert hasattr(PrioridadCatSalut, 'REFERENCIA_P2')
        assert hasattr(PrioridadCatSalut, 'REFERENCIA_P3')
    
    def test_tiempos_maximos_correctos(self):
        """Verificar tiempos máximos según normativa"""
        from config import TIEMPOS_MAXIMOS_ESPERA, PrioridadCatSalut
        
        assert TIEMPOS_MAXIMOS_ESPERA[PrioridadCatSalut.ONCOLOGICO_PRIORITARIO] == 45
        assert TIEMPOS_MAXIMOS_ESPERA[PrioridadCatSalut.ONCOLOGICO_ESTANDAR] == 60
        assert TIEMPOS_MAXIMOS_ESPERA[PrioridadCatSalut.REFERENCIA_P1] == 90
        assert TIEMPOS_MAXIMOS_ESPERA[PrioridadCatSalut.REFERENCIA_P2] == 180
    
    def test_quirofanos_configurados(self):
        """Verificar configuración de 8 quirófanos"""
        from config import QUIROFANOS_DEFAULT
        
        assert len(QUIROFANOS_DEFAULT) == 8
        for q in QUIROFANOS_DEFAULT:
            assert q.horario_inicio == 8 * 60  # 08:00
            assert q.horario_fin == 15 * 60    # 15:00
    
    def test_pesos_optimizacion_validos(self):
        """Verificar que los pesos suman 1.0"""
        from config import PesosOptimizacion
        
        pesos = PesosOptimizacion()
        assert abs(pesos.peso_prioridad_clinica + pesos.peso_eficiencia_operativa - 1.0) < 0.001


class TestModels:
    """Tests para models.py"""
    
    def test_crear_paciente(self):
        """Verificar creación de paciente"""
        from models import Paciente, ClaseASA
        from datetime import date
        
        paciente = Paciente(
            id="PAC-001",
            nombre="Test Patient",
            fecha_nacimiento=date(1980, 1, 1),
            sexo="M",
            numero_historia="HC123456",
            clase_asa=ClaseASA.ASA_II
        )
        
        assert paciente.id == "PAC-001"
        assert paciente.edad > 40
        assert not paciente.es_pediatrico
    
    def test_solicitud_cirugia_fecha_limite(self):
        """Verificar cálculo automático de fecha límite"""
        from models import Paciente, SolicitudCirugia, ClaseASA
        from config import PrioridadCatSalut, CATALOGO_INTERVENCIONES
        from datetime import date, timedelta
        
        paciente = Paciente(
            id="PAC-002",
            nombre="Test",
            fecha_nacimiento=date(1970, 1, 1),
            sexo="F",
            numero_historia="HC999",
            clase_asa=ClaseASA.ASA_I
        )
        
        solicitud = SolicitudCirugia(
            id="SOL-001",
            paciente=paciente,
            tipo_intervencion=CATALOGO_INTERVENCIONES["COL_LAP"],
            fecha_indicacion=date.today() - timedelta(days=30),
            prioridad=PrioridadCatSalut.REFERENCIA_P2
        )
        
        assert solicitud.dias_en_espera == 30
        assert not solicitud.esta_fuera_plazo  # P2 = 180 días
    
    def test_programa_diario_sin_conflictos(self):
        """Verificar que no se permiten conflictos de horario"""
        from models import ProgramaDiario, CirugiaProgramada, SolicitudCirugia, Paciente, Cirujano, ClaseASA
        from config import CATALOGO_INTERVENCIONES, PrioridadCatSalut, Especialidad
        from datetime import date
        
        # Crear paciente y cirujano de prueba
        paciente = Paciente("P1", "Test", date(1980,1,1), "M", "HC1", ClaseASA.ASA_I)
        cirujano = Cirujano("C1", "Dr. Test", Especialidad.CIRUGIA_DIGESTIVA)
        
        solicitud = SolicitudCirugia(
            id="S1", paciente=paciente,
            tipo_intervencion=CATALOGO_INTERVENCIONES["COL_LAP"],
            fecha_indicacion=date.today(),
            prioridad=PrioridadCatSalut.REFERENCIA_P2
        )
        
        programa = ProgramaDiario(fecha=date.today())
        
        # Primera cirugía
        c1 = CirugiaProgramada(
            id="CIR1", solicitud=solicitud,
            fecha=date.today(), hora_inicio=480,  # 08:00
            duracion_programada_min=60, quirofano_id=1, cirujano=cirujano
        )
        assert programa.agregar_cirugia(c1) == True
        
        # Segunda cirugía en conflicto (mismo quirófano, hora solapada)
        c2 = CirugiaProgramada(
            id="CIR2", solicitud=solicitud,
            fecha=date.today(), hora_inicio=500,  # 08:20
            duracion_programada_min=60, quirofano_id=1, cirujano=cirujano
        )
        assert programa.agregar_cirugia(c2) == False  # Debe rechazarla


class TestSyntheticData:
    """Tests para synthetic_data.py"""
    
    def test_generar_paciente(self):
        """Verificar generación de paciente sintético"""
        from synthetic_data import GeneradorDatosSinteticos
        
        generador = GeneradorDatosSinteticos(seed=42)
        paciente = generador.generar_paciente()
        
        assert paciente.id.startswith("PAC-")
        assert paciente.nombre
        assert paciente.sexo in ['M', 'F']
        assert 18 <= paciente.edad <= 95
    
    def test_generar_lista_espera(self):
        """Verificar generación de lista de espera"""
        from synthetic_data import GeneradorDatosSinteticos
        
        generador = GeneradorDatosSinteticos(seed=42)
        cirujanos = generador.generar_equipo_cirujanos()
        lista = generador.generar_lista_espera(cirujanos, n_solicitudes=50)
        
        assert len(lista) == 50
        assert all(s.id.startswith("SOL-") for s in lista)
        # Debe estar ordenada por score
        scores = [s.score_clinico for s in lista]
        assert scores == sorted(scores, reverse=True)
    
    def test_generar_historico(self):
        """Verificar generación de histórico"""
        from synthetic_data import GeneradorDatosSinteticos
        
        generador = GeneradorDatosSinteticos(seed=42)
        cirujanos = generador.generar_equipo_cirujanos()
        historico = generador.generar_historico_cirugias(cirujanos, dias=30, cirugias_por_dia=10)
        
        assert len(historico) > 0
        assert 'fecha' in historico.columns
        assert 'duracion_real' in historico.columns
        assert 'tipo_intervencion' in historico.columns


class TestOptimizer:
    """Tests para optimizer.py"""
    
    def test_optimizador_basico(self):
        """Verificar que el optimizador produce un resultado"""
        from synthetic_data import GeneradorDatosSinteticos
        from optimizer import OptimizadorQuirurgico
        from datetime import date, timedelta
        
        # Generar datos
        generador = GeneradorDatosSinteticos(seed=42)
        cirujanos = generador.generar_equipo_cirujanos()
        lista_espera = generador.generar_lista_espera(cirujanos, n_solicitudes=30)
        
        # Optimizar
        optimizador = OptimizadorQuirurgico()
        resultado = optimizador.optimizar(
            solicitudes=lista_espera,
            cirujanos=cirujanos,
            fecha_inicio=date.today() + timedelta(days=1),
            fecha_fin=date.today() + timedelta(days=5)
        )
        
        assert resultado.cirugias_programadas > 0
        assert resultado.score_total > 0
        assert resultado.tiempo_ejecucion_seg > 0


class TestConstraintLearning:
    """Tests para constraint_learning.py"""
    
    def test_aprender_restricciones(self):
        """Verificar aprendizaje de restricciones"""
        from synthetic_data import GeneradorDatosSinteticos
        from constraint_learning import AprendizajeRestricciones
        
        # Generar histórico
        generador = GeneradorDatosSinteticos(seed=42)
        cirujanos = generador.generar_equipo_cirujanos()
        historico = generador.generar_historico_cirugias(cirujanos, dias=180, cirugias_por_dia=20)
        
        # Aprender
        aprendizaje = AprendizajeRestricciones()
        restricciones = aprendizaje.analizar_historico(historico)
        
        assert len(restricciones) > 0
        # Debe encontrar al menos preferencias de quirófano
        tipos = [r.tipo for r in restricciones]
        assert any('preferencia' in t or 'asignacion' in t for t in tipos)


class TestIntegracion:
    """Tests de integración del sistema completo"""
    
    def test_flujo_completo(self):
        """Verificar flujo completo: datos -> aprendizaje -> optimización"""
        from main import ProgramadorQuirurgico
        
        programador = ProgramadorQuirurgico(seed=42)
        programador.inicializar_datos_sinteticos(
            n_solicitudes=50,
            dias_historico=90
        )
        
        assert len(programador.lista_espera) == 50
        assert len(programador.cirujanos) > 0
        assert len(programador.restricciones_aprendidas) > 0
        
        # Optimizar
        resultado = programador.optimizar_programa(horizonte_dias=5)
        
        assert resultado.cirugias_programadas > 0
        assert resultado.score_total > 0


# Ejecutar si se llama directamente
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
