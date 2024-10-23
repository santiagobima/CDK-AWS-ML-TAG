import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_parameter", type=str)
parser.add_argument("--name", type=str)
args = parser.parse_args()

print(f"Hello {args.config_parameter}!")  # Esto debería imprimir "Cloud Developer" u otro valor que hayas pasado
