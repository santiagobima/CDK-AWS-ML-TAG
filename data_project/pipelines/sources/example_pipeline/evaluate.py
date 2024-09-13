import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_parameter", type=str)
parser.add_argument("--name", type=str)
args = parser.parse_args()

print(f"Hello {args.config_parameter}!")
name = args.name
name = {
    "name": "santiago",
    "age": 36,
}