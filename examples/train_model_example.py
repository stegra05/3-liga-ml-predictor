"""
Example: Train a CatBoost Model for 3. Liga Match Prediction
Demonstrates how to use the dataset for machine learning
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Load the pre-split datasets"""
    print("Loading datasets...")

    train = pd.read_csv('../data/processed/3liga_ml_dataset_train.csv')
    val = pd.read_csv('../data/processed/3liga_ml_dataset_val.csv')
    test = pd.read_csv('../data/processed/3liga_ml_dataset_test.csv')

    print(f"Train set: {len(train)} matches")
    print(f"Validation set: {len(val)} matches")
    print(f"Test set: {len(test)} matches")

    return train, val, test


def prepare_features(train, val, test):
    """
    Prepare feature sets for modeling

    Uses research-backed features for gradient boosting:
    - Pi-ratings (most important)
    - Elo ratings
    - Form metrics
    - Optionally: betting odds
    """

    # Core rating features (always available)
    core_features = [
        'home_elo', 'away_elo', 'elo_diff',
        'home_pi', 'away_pi', 'pi_diff',
        'home_points_l5', 'away_points_l5', 'form_diff_l5',
        'home_points_l10', 'away_points_l10',
        'home_goals_scored_l5', 'home_goals_conceded_l5',
        'away_goals_scored_l5', 'away_goals_conceded_l5',
        'goal_diff_l5'
    ]

    # Betting odds (not always available, handle missing)
    odds_features = [
        'implied_prob_home', 'implied_prob_draw', 'implied_prob_away'
    ]

    # Context features
    context_features = [
        'day_of_week', 'month', 'matchday'
    ]

    # Combine features
    features = core_features + context_features

    # Add odds if available (check coverage)
    odds_coverage = train[odds_features[0]].notna().mean()
    if odds_coverage > 0.5:  # If >50% coverage
        print(f"Including betting odds features ({odds_coverage*100:.1f}% coverage)")
        features += odds_features
    else:
        print(f"Excluding betting odds features (only {odds_coverage*100:.1f}% coverage)")

    # Target: 3-class classification
    # 0 = Away win, 1 = Draw, 2 = Home win
    target = 'target_multiclass'

    # Prepare datasets
    X_train = train[features].fillna(train[features].median())
    y_train = train[target]

    X_val = val[features].fillna(train[features].median())  # Use train medians
    y_val = val[target]

    X_test = test[features].fillna(train[features].median())
    y_test = test[target]

    print(f"\nFeatures: {len(features)}")
    print("Feature list:", features)

    return X_train, y_train, X_val, y_val, X_test, y_test, features


def train_model(X_train, y_train, X_val, y_val):
    """
    Train CatBoost model

    CatBoost is chosen because research shows gradient boosting
    with pi-ratings achieves state-of-the-art performance (55.82% accuracy)
    """

    print("\n" + "="*60)
    print("Training CatBoost Model")
    print("="*60)

    # Create CatBoost pools
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    # Model parameters
    params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'use_best_model': True
    }

    # Train
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        plot=False
    )

    return model


def evaluate_model(model, X_test, y_test, features):
    """Evaluate model performance"""

    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Compare to baseline
    baseline_accuracy = y_test.value_counts().max() / len(y_test)
    print(f"Baseline (always predict most common): {baseline_accuracy:.4f}")
    print(f"Improvement over baseline: +{(accuracy - baseline_accuracy)*100:.2f}%")

    # Classification report
    print("\nClassification Report:")
    target_names = ['Away Win', 'Draw', 'Home Win']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("           Away  Draw  Home")
    for i, row_label in enumerate(['Away Win', 'Draw', 'Home Win']):
        print(f"{row_label:10s}", "  ".join(f"{val:5d}" for val in cm[i]))

    # Feature importance
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    return accuracy, importance_df


def plot_results(importance_df, save_path='../data/processed/'):
    """Plot feature importance"""

    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)

    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title('Top 15 Feature Importance for 3. Liga Match Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    output_file = f'{save_path}feature_importance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plot saved: {output_file}")


def main():
    """Main execution"""

    print("="*60)
    print("3. LIGA MATCH PREDICTION WITH CATBOOST")
    print("="*60)

    # Load data
    train, val, test = load_data()

    # Prepare features
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_features(train, val, test)

    # Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    accuracy, importance_df = evaluate_model(model, X_test, y_test, features)

    # Plot results
    try:
        plot_results(importance_df)
    except Exception as e:
        print(f"Could not create plot: {e}")

    # Save model
    model_path = '../data/processed/3liga_catboost_model.cbm'
    model.save_model(model_path)
    print(f"\nModel saved: {model_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("\nNext steps:")
    print("  1. Tune hyperparameters for better performance")
    print("  2. Try ensemble methods (combine multiple models)")
    print("  3. Add more features when available")
    print("  4. Experiment with different targets (goals, etc.)")


if __name__ == "__main__":
    main()
