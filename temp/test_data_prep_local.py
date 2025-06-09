import logging
import os
import sys
import pandas as pd

# Añadir la raíz del proyecto al PYTHONPATH
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_PATH)

# ✅ Ahora sí importa después de ajustar sys.path
from pipelines.lead_conversion_rate.common.utils.data_prep import get_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        stage = os.getenv('CDK_ENV', 'dev')
        logger.info(f"🚀 Ejecutando `get_features` con stage={stage}")
        
        df = get_features(stage)

        if df is None:
            logger.error("❌ El DataFrame devuelto es None.")
        elif df.empty:
            logger.warning("⚠️ El DataFrame está vacío.")
        else:
            logger.info(f'✅ DataFrame generado con shape: {df.shape}')
            df.head(10).to_csv('test_output_local.csv', index=False)
            logger.info("📁 Primeras filas guardadas como 'test_output_local.csv'")
    
    except Exception as e:
        logger.exception('❌ Error al ejecutar el test local')
