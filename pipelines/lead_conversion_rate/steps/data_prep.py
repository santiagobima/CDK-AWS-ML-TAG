import os
import sys
import subprocess
import pandas as pd
import numpy as np

# Instalación del paquete en ejecución dentro de SageMaker
if os.path.exists("/opt/ml/processing/source_code"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "/opt/ml/processing/source_code"])
    sys.path.insert(0, "/opt/ml/processing/source_code")






import os
import sys
import logging
import argparse
import pandas as pd
from pipelines.lead_conversion_rate.common.utils.transformers import (
    BooleanTransformer, ReplaceTransformer, CountryCodeTransformer,
    LocationTransformer, EnrichmentTransformer, FillnaTransformer,
    DealCookingStateTransformer, ChangeTypeTransformer,
    YearsOfExperienceTransformer, ScalerTransformer,
    CombineProfileTransformer, OneHotEncodeTransformer,
    OneHotEncodeMultipleChoicesTransformer,
    CalculateTimeSinceTransformer, DropColumnsTransformer,
    FeatureNamesSanitizerTransformer, PreprocessSummary
)
from sklearn.pipeline import Pipeline
from pipelines.lead_conversion_rate.common.constants import (
    BOOLEAN_COLUMNS, REPLACE_DICT, ENRICHMENT_COLUMNS,
    FILLNA_VALUES, TYPE_DICT, COLUMNS_TO_SCALE,
    ONEHOT_COLUMNS, MULTIPLE_CATEGORIES, TIME_FIELDS
)

from pipelines.lead_conversion_rate.model.utls.utls import config


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocessing_pipeline(prediction=True):
    steps = [
        ('boolean', BooleanTransformer(columns=BOOLEAN_COLUMNS)),
        ('replace', ReplaceTransformer(replace_dict=REPLACE_DICT)),
        ('convert_country', CountryCodeTransformer()),
        ('location', LocationTransformer()),
        ('enrich', EnrichmentTransformer(column_pairs=ENRICHMENT_COLUMNS)),
        ('fillna', FillnaTransformer(fill_values=FILLNA_VALUES)),
        ("deal_cooking", DealCookingStateTransformer()),
        ('replace_review_analysis', ReplaceTransformer(replace_dict={'review_analysis': {'no': '0', 'yes': '1'}})),
        ("type", ChangeTypeTransformer(TYPE_DICT)),
        ("years_of_exp_type", YearsOfExperienceTransformer()),
        ('scale', ScalerTransformer(columns=COLUMNS_TO_SCALE)),
        ('combine_profile', CombineProfileTransformer()),
        ('onehot_encode', OneHotEncodeTransformer(columns=ONEHOT_COLUMNS)),
        ('onehot_encode_multichoice', OneHotEncodeMultipleChoicesTransformer(columns=MULTIPLE_CATEGORIES)),
        ('time_since', CalculateTimeSinceTransformer(time_fields=TIME_FIELDS)),
        ('drop_columns', DropColumnsTransformer()),
        ("type_phase2", ChangeTypeTransformer()),
        ('name_sanitizer', FeatureNamesSanitizerTransformer())
    ]
    if not prediction:
        steps.append(('Summary', PreprocessSummary()))
    return Pipeline(steps=steps, verbose=True)






def main(input_path, output_path):
    if not os.path.exists(input_path):
        logger.error(f"❌ Archivo de entrada no encontrado: {input_path}")
        sys.exit(1)

    logger.info(f"📥 Cargando datos desde: {input_path}")
    baseline_df = pd.read_pickle(input_path)

    logger.info("⚙️ Aplicando pipeline de preprocesamiento...")
    processed_df = preprocessing_pipeline(prediction=False).fit_transform(baseline_df)

    if isinstance(processed_df, np.ndarray):
        processed_df = pd.DataFrame(processed_df)

    

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"💾 Guardando archivo procesado en: {output_path}")
    processed_df.to_pickle(output_path)
    logger.info("✅ Procesamiento completado exitosamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_input = "/opt/ml/processing/retrieve/train.pkl" if os.path.exists("/opt/ml/processing/input") else "pipelines/lead_conversion_rate/model/pickles/train.pkl"
    default_output = "/opt/ml/processing/output/baseline_features_raw.pkl" if os.path.exists("/opt/ml/processing/input") else "pipelines/lead_conversion_rate/model/pickles/baseline_features_raw.pkl"

    parser.add_argument("--input_path", type=str, default=default_input)
    parser.add_argument("--output_path", type=str, default=default_output)

    args = parser.parse_args()
    main(args.input_path, args.output_path)
    
    
    
    
    
       