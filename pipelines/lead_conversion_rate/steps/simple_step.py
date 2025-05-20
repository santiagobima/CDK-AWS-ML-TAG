# simple_step.py

import argparse
import logging
import sys
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


logger.info("This step doesn't require S3 inputs or outputs.")
logger.info("Processing SIMPLE_STEP completed successfully!")

try:
    input_contents = os.listdir('/opt/ml/processing/input')
    logger.info(f"📁 Input directory contents: {input_contents}")
except FileNotFoundError:
    input_contents = []
    logger.warning("⚠️ Input directory not found.")

output_path = "/opt/ml/processing/output/log_check.txt"
with open(output_path, "w") as f:
    f.write("✅ simple_step.py completed successfully.\n")
    f.write(f"📁 Input contents: {input_contents}\n")

logger.info(f"📝 Output written to {output_path}")

    