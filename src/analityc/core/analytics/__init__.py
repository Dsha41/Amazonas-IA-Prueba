"""
analytics - Modulos de analitica retail en tiempo real.

Submodulos:
    demographics       - Clasificacion de genero y rango de edad
    people_counter     - Conteo unico de personas
    attendance_tracker - Deteccion de atencion vendedor-cliente
    seller_efficiency  - Metricas de eficiencia y premio horario
    stock_monitor      - Monitoreo de productos por ROI
    config             - Parametros configurables
"""

from .config import AnalyticsConfig
from .demographics import DemographicsClassifier
from .people_counter import PeopleCounter
from .attendance_tracker import AttendanceTracker
from .seller_efficiency import SellerEfficiency
from .stock_monitor import StockMonitor
