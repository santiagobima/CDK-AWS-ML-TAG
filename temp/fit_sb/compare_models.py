import joblib
import pandas as pd
import numpy as np
import json

def load_model(model_path):
    return joblib.load(model_path)

def load_features_json(features_path):
    with open(features_path, "r") as f:
        features_dict = json.load(f)
    if isinstance(features_dict, dict):
        features = list(features_dict.keys())
        feature_types = features_dict
    else:
        features = features_dict
        feature_types = {}
    return features, feature_types

def compare_features(features_sb, features_dp):
    print("\n=== Feature Comparison ===")
    print(f"SB ({len(features_sb)} features):")
    print(f"DP ({len(features_dp)} features):")
    diff_sb = set(features_sb) - set(features_dp)
    diff_dp = set(features_dp) - set(features_sb)
    if not diff_sb and not diff_dp:
        print("✅ Features are identical!")
    else:
        if diff_sb:
            print(f"❌ Features only in SB ({len(diff_sb)}): {sorted(diff_sb)}")
        if diff_dp:
            print(f"❌ Features only in DP ({len(diff_dp)}): {sorted(diff_dp)}")
            
            
            

# Print detailed differences between feature sets
def print_feature_differences(features_sb, features_dp):
    print("\n=== Feature Differences Breakdown ===")
    set_sb = set(features_sb)
    set_dp = set(features_dp)

    only_in_sb = sorted(set_sb - set_dp)
    only_in_dp = sorted(set_dp - set_sb)

    print(f"\nFeatures only in SB ({len(only_in_sb)}):")
    for feat in only_in_sb:
        print(f"  - {feat}")

    print(f"\nFeatures only in DP ({len(only_in_dp)}):")
    for feat in only_in_dp:
        print(f"  - {feat}")


def compare_models(model_sb, model_dp):
    print("\n=== Model Comparison ===")
    print(f"Type SB: {type(model_sb)}")
    print(f"Type DP: {type(model_dp)}")
    print("\nSB model parameters:")
    print(getattr(model_sb, 'get_params', lambda: 'N/A')())
    print("\nDP model parameters:")
    print(getattr(model_dp, 'get_params', lambda: 'N/A')())

if __name__ == "__main__":
    # Paths
    model_sb_path = "temp/fit_sb/model_SB.joblib"
    model_dp_path = "temp/fit_sb/model_DP.joblib"
    features_sb_path = "temp/fit_sb/model_SB.json"
    features_dp_path = "temp/fit_sb/model_DP.json"

    # Load models and feature metadata
    model_sb = load_model(model_sb_path)
    model_dp = load_model(model_dp_path)
    features_sb, feature_types_sb = load_features_json(features_sb_path)
    features_dp, feature_types_dp = load_features_json(features_dp_path)

    # Compare features and model structure
    compare_features(features_sb, features_dp)
    print_feature_differences(features_sb, features_dp)
    compare_models(model_sb, model_dp)
    
    