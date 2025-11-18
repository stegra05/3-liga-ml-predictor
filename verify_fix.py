
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from liga_predictor import evaluation, config

def test_feature_selection():
    print("Testing feature selection in evaluation.py...")
    
    # Create a mock DataFrame with valid features AND post-match stats
    data = {
        'match_id': [1, 2],
        'target_multiclass': [0, 1],
        'home_elo': [1500, 1600],  # Valid feature
        'post_home_shots': [10, 5], # Post-match stat (should be excluded)
        'home_shots': [10, 5],      # Old name (should be excluded if present)
        'random_col': [1, 2]        # Random column (should be excluded)
    }
    df = pd.DataFrame(data)
    
    # Run preparation
    X, y = evaluation.prepare_features_target(df)
    
    # Check features
    print(f"Features selected: {X.columns.tolist()}")
    
    if 'home_elo' not in X.columns:
        print("FAIL: Valid feature 'home_elo' missing!")
        return False
        
    if 'post_home_shots' in X.columns:
        print("FAIL: Post-match stat 'post_home_shots' included!")
        return False
        
    if 'home_shots' in X.columns:
        print("FAIL: Old post-match stat 'home_shots' included!")
        return False
        
    if 'random_col' in X.columns:
        print("FAIL: Random column 'random_col' included!")
        return False
        
    print("SUCCESS: Only valid features selected.")
    return True

if __name__ == "__main__":
    if test_feature_selection():
        sys.exit(0)
    else:
        sys.exit(1)
