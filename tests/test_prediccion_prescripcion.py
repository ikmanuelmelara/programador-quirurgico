"""
Tests para E1: Separación Predicción/Prescripción
=================================================

Verifica:
1. PredictorDemanda.obtener_flujo_semanal() devuelve datos correctos
2. OptimizadorPrescriptivo funciona con distintos objetivos
3. Integración predicción -> prescripción

Ejecutar con: pytest tests/test_prediccion_prescripcion.py -v
"""

import pytest
import sys
import os
import numpy as np

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def predictor_entrenado():
    """Crea un predictor de demanda entrenado con datos sintéticos"""
    from predictor_demanda import (
        PredictorDemanda, GeneradorHistoricoMovimientos
    )
    from datetime import date, timedelta

    generador = GeneradorHistoricoMovimientos()
    historico = generador.generar_historico(
        fecha_inicio=date.today() - timedelta(days=120),
        fecha_fin=date.today() - timedelta(days=1)
    )
    predictor = PredictorDemanda(historico)
    predictor.entrenar()
    return predictor


@pytest.fixture
def simulador_whatif():
    """Crea un simulador What-If con configuración de ejemplo"""
    from simulador_whatif import SimuladorWhatIf

    config_sesiones = {}
    especialidades = [
        'CIRUGIA_GENERAL', 'CIRUGIA_DIGESTIVA', 'UROLOGIA',
        'GINECOLOGIA', 'CIRUGIA_MAMA', 'CIRUGIA_COLORRECTAL',
        'CIRUGIA_VASCULAR', 'CIRUGIA_BARIATRICA'
    ]
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
    for q in range(1, 9):
        config_sesiones[q] = {}
        for dia in dias:
            esp = especialidades[(q - 1) % len(especialidades)]
            config_sesiones[q][dia] = {'Mañana': esp, 'Tarde': 'LIBRE'}

    tasas_entrada = {
        'CIRUGIA_GENERAL': 15, 'CIRUGIA_DIGESTIVA': 10,
        'UROLOGIA': 12, 'GINECOLOGIA': 10, 'CIRUGIA_MAMA': 8,
        'CIRUGIA_COLORRECTAL': 6, 'CIRUGIA_VASCULAR': 7,
        'CIRUGIA_BARIATRICA': 4,
    }

    return SimuladorWhatIf(
        configuracion_sesiones=config_sesiones,
        lista_espera_actual=500,
        fuera_plazo_actual=50,
        tasas_entrada=tasas_entrada,
    )


# =============================================================================
# TESTS: PredictorDemanda.obtener_flujo_semanal()
# =============================================================================

class TestObtenerFlujoSemanal:
    """Tests para el nuevo método obtener_flujo_semanal()"""

    def test_devuelve_dict_no_vacio(self, predictor_entrenado):
        """El flujo semanal debe ser un dict con al menos una especialidad"""
        flujo = predictor_entrenado.obtener_flujo_semanal()
        assert isinstance(flujo, dict)
        assert len(flujo) > 0

    def test_estructura_por_especialidad(self, predictor_entrenado):
        """Cada especialidad debe tener las claves esperadas"""
        flujo = predictor_entrenado.obtener_flujo_semanal()
        claves_esperadas = {
            'entradas_media', 'entradas_std',
            'salidas_media', 'salidas_std', 'balance'
        }
        for esp, datos in flujo.items():
            assert claves_esperadas.issubset(datos.keys()), \
                f"Faltan claves en {esp}: {claves_esperadas - datos.keys()}"

    def test_valores_no_negativos(self, predictor_entrenado):
        """Las medias y desviaciones no deben ser negativas"""
        flujo = predictor_entrenado.obtener_flujo_semanal()
        for esp, datos in flujo.items():
            assert datos['entradas_media'] >= 0, f"{esp}: entradas_media negativa"
            assert datos['entradas_std'] >= 0, f"{esp}: entradas_std negativa"
            assert datos['salidas_media'] >= 0, f"{esp}: salidas_media negativa"
            assert datos['salidas_std'] >= 0, f"{esp}: salidas_std negativa"

    def test_balance_coherente(self, predictor_entrenado):
        """El balance debe ser entradas - salidas"""
        flujo = predictor_entrenado.obtener_flujo_semanal()
        for esp, datos in flujo.items():
            balance_esperado = datos['entradas_media'] - datos['salidas_media']
            assert abs(datos['balance'] - balance_esperado) < 0.01, \
                f"{esp}: balance incoherente"

    def test_entrena_automaticamente_si_necesario(self):
        """Si no está entrenado, obtener_flujo_semanal() debe entrenar"""
        from predictor_demanda import (
            PredictorDemanda, GeneradorHistoricoMovimientos
        )
        from datetime import date, timedelta

        generador = GeneradorHistoricoMovimientos()
        historico = generador.generar_historico(
            fecha_inicio=date.today() - timedelta(days=120),
            fecha_fin=date.today() - timedelta(days=1)
        )
        predictor = PredictorDemanda(historico)
        assert not predictor.modelo_entrenado

        flujo = predictor.obtener_flujo_semanal()
        assert predictor.modelo_entrenado
        assert len(flujo) > 0


# =============================================================================
# TESTS: OptimizadorPrescriptivo
# =============================================================================

class TestOptimizadorPrescriptivo:
    """Tests para el nuevo OptimizadorPrescriptivo"""

    def test_prescribir_equilibrio_flujo(self, simulador_whatif, predictor_entrenado):
        """Prescripción con objetivo EQUILIBRAR_FLUJO debe devolver resultado válido"""
        from simulador_whatif import ObjetivoPrescripcion, TipoObjetivo

        flujo = predictor_entrenado.obtener_flujo_semanal()
        simulador_whatif.set_flujo_predicho(flujo)

        objetivo = ObjetivoPrescripcion(
            tipo=TipoObjetivo.EQUILIBRAR_FLUJO,
            semanas=12,
            confianza=0.8,
        )
        resultado = simulador_whatif.prescribir(objetivo)

        assert resultado is not None
        assert isinstance(resultado.sesiones_recomendadas, dict)
        assert resultado.lista_final_esperada >= 0
        assert resultado.fp_final_esperado >= 0
        assert len(resultado.explicacion) > 0

    def test_prescribir_eliminar_fp(self, simulador_whatif, predictor_entrenado):
        """Prescripción ELIMINAR_FUERA_PLAZO debe recomendar sesiones extra"""
        from simulador_whatif import ObjetivoPrescripcion, TipoObjetivo

        flujo = predictor_entrenado.obtener_flujo_semanal()
        simulador_whatif.set_flujo_predicho(flujo)

        objetivo = ObjetivoPrescripcion(
            tipo=TipoObjetivo.ELIMINAR_FUERA_PLAZO,
            semanas=12,
            confianza=0.8,
        )
        resultado = simulador_whatif.prescribir(objetivo)

        # Con FP=50, debería recomendar sesiones extra
        total_extra = sum(resultado.sesiones_recomendadas.values())
        assert total_extra > 0, "Debería recomendar sesiones extra para eliminar FP"

    def test_prescribir_reducir_fp_parcial(self, simulador_whatif, predictor_entrenado):
        """Reducir FP al 50% debe requerir menos sesiones que eliminar al 100%"""
        from simulador_whatif import ObjetivoPrescripcion, TipoObjetivo

        flujo = predictor_entrenado.obtener_flujo_semanal()
        simulador_whatif.set_flujo_predicho(flujo)

        obj_100 = ObjetivoPrescripcion(
            tipo=TipoObjetivo.REDUCIR_FUERA_PLAZO,
            reduccion_fp_pct=100.0,
            semanas=12,
        )
        obj_50 = ObjetivoPrescripcion(
            tipo=TipoObjetivo.REDUCIR_FUERA_PLAZO,
            reduccion_fp_pct=50.0,
            semanas=12,
        )

        res_100 = simulador_whatif.prescribir(obj_100)
        res_50 = simulador_whatif.prescribir(obj_50)

        total_100 = sum(res_100.sesiones_recomendadas.values())
        total_50 = sum(res_50.sesiones_recomendadas.values())

        assert total_50 <= total_100, \
            f"Reducir 50% ({total_50}) no debería requerir más que 100% ({total_100})"

    def test_prescribir_reducir_lista(self, simulador_whatif, predictor_entrenado):
        """REDUCIR_LISTA debe devolver resultado con simulación"""
        from simulador_whatif import ObjetivoPrescripcion, TipoObjetivo

        flujo = predictor_entrenado.obtener_flujo_semanal()
        simulador_whatif.set_flujo_predicho(flujo)

        objetivo = ObjetivoPrescripcion(
            tipo=TipoObjetivo.REDUCIR_LISTA,
            reduccion_lista_pct=20.0,
            semanas=12,
        )
        resultado = simulador_whatif.prescribir(objetivo)

        assert resultado.simulacion is not None, "Debería incluir simulación"
        assert len(resultado.comparacion_capacidad) > 0

    def test_max_sesiones_respetado(self, simulador_whatif, predictor_entrenado):
        """El límite max_sesiones_extra debe respetarse"""
        from simulador_whatif import ObjetivoPrescripcion, TipoObjetivo

        flujo = predictor_entrenado.obtener_flujo_semanal()
        simulador_whatif.set_flujo_predicho(flujo)

        max_ses = 3
        objetivo = ObjetivoPrescripcion(
            tipo=TipoObjetivo.ELIMINAR_FUERA_PLAZO,
            semanas=12,
            max_sesiones_extra=max_ses,
        )
        resultado = simulador_whatif.prescribir(objetivo)

        for esp, n in resultado.sesiones_recomendadas.items():
            assert n <= max_ses, \
                f"{esp}: {n} sesiones excede el máximo {max_ses}"

    def test_error_sin_flujo_predicho(self, simulador_whatif):
        """Prescribir sin haber conectado el flujo debe lanzar RuntimeError"""
        from simulador_whatif import ObjetivoPrescripcion, TipoObjetivo

        objetivo = ObjetivoPrescripcion(tipo=TipoObjetivo.EQUILIBRAR_FLUJO)

        with pytest.raises(RuntimeError, match="predicción"):
            simulador_whatif.prescribir(objetivo)

    def test_comparacion_capacidad_completa(self, simulador_whatif, predictor_entrenado):
        """La comparación de capacidad debe tener las claves esperadas"""
        from simulador_whatif import ObjetivoPrescripcion, TipoObjetivo

        flujo = predictor_entrenado.obtener_flujo_semanal()
        simulador_whatif.set_flujo_predicho(flujo)

        objetivo = ObjetivoPrescripcion(
            tipo=TipoObjetivo.EQUILIBRAR_FLUJO,
            semanas=12,
        )
        resultado = simulador_whatif.prescribir(objetivo)

        for esp, comp in resultado.comparacion_capacidad.items():
            assert 'actual' in comp, f"{esp}: falta 'actual'"
            assert 'recomendada' in comp, f"{esp}: falta 'recomendada'"
            assert 'sesiones_extra' in comp, f"{esp}: falta 'sesiones_extra'"
            assert 'entradas' in comp, f"{esp}: falta 'entradas'"


# =============================================================================
# TESTS: Integración predicción -> prescripción
# =============================================================================

class TestIntegracionPrediccionPrescripcion:
    """Tests de integración del flujo completo"""

    def test_flujo_completo_prediccion_a_prescripcion(self):
        """Verifica el flujo completo: entrenar -> predecir -> prescribir"""
        from predictor_demanda import (
            PredictorDemanda, GeneradorHistoricoMovimientos
        )
        from simulador_whatif import (
            SimuladorWhatIf, ObjetivoPrescripcion, TipoObjetivo
        )
        from datetime import date, timedelta

        # 1. Generar datos y entrenar predictor
        generador = GeneradorHistoricoMovimientos()
        historico = generador.generar_historico(
            fecha_inicio=date.today() - timedelta(days=120),
            fecha_fin=date.today() - timedelta(days=1)
        )
        predictor = PredictorDemanda(historico)
        predictor.entrenar()

        # 2. Obtener flujo semanal (interfaz predicción -> prescripción)
        flujo = predictor.obtener_flujo_semanal()
        assert len(flujo) > 0

        # 3. Crear simulador y conectar flujo
        config_sesiones = {}
        especialidades = list(flujo.keys())[:8]
        dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
        for q in range(1, 9):
            config_sesiones[q] = {}
            for dia in dias:
                esp = especialidades[q % len(especialidades)]
                config_sesiones[q][dia] = {'Mañana': esp, 'Tarde': 'LIBRE'}

        tasas_entrada = {esp: datos['entradas_media']
                         for esp, datos in flujo.items()}

        sim = SimuladorWhatIf(
            configuracion_sesiones=config_sesiones,
            lista_espera_actual=500,
            fuera_plazo_actual=50,
            tasas_entrada=tasas_entrada,
        )

        sim.set_flujo_predicho(flujo)

        # 4. Prescribir
        objetivo = ObjetivoPrescripcion(
            tipo=TipoObjetivo.EQUILIBRAR_FLUJO,
            semanas=12,
        )
        resultado = sim.prescribir(objetivo)

        # 5. Verificar resultado
        assert resultado is not None
        assert len(resultado.explicacion) > 0
        assert resultado.flujo_entrada == tasas_entrada

    def test_tipos_objetivo_exportados(self):
        """Verificar que los nuevos tipos están exportados correctamente"""
        from simulador_whatif import (
            ObjetivoPrescripcion, TipoObjetivo,
            ResultadoPrescripcion, OptimizadorPrescriptivo
        )

        # Verificar enum
        assert hasattr(TipoObjetivo, 'ELIMINAR_FUERA_PLAZO')
        assert hasattr(TipoObjetivo, 'REDUCIR_FUERA_PLAZO')
        assert hasattr(TipoObjetivo, 'EQUILIBRAR_FLUJO')
        assert hasattr(TipoObjetivo, 'REDUCIR_LISTA')
