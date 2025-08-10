#!/usr/bin/env python3
"""
Improved Housing Prices Prediction Model
Addresses overfitting issues in the original Random Forest model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load training and test datasets"""
    print("Loading datasets...")
    
    # Use the current script directory to find data files
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        dataset_train = pd.read_csv(os.path.join(script_dir, "train.csv"))
        dataset_test = pd.read_csv(os.path.join(script_dir, "test.csv"))
    except FileNotFoundError:
        print("Error: Please make sure 'train.csv' and 'test.csv' are in the same directory as the script")
        print(f"Script location: {script_dir}")
        return None, None
    
    print(f"Training dataset shape: {dataset_train.shape}")
    print(f"Test dataset shape: {dataset_test.shape}")
    return dataset_train, dataset_test

def analyze_missing_values(dataset_train):
    """Analyze and display missing values"""
    print("\n=== MISSING VALUES ANALYSIS ===")
    missing_values = dataset_train.isnull().sum().sort_values(ascending=False)
    print("Missing values in training data:")
    print(missing_values[missing_values > 0])
    
    # Visualize missing values
    plt.figure(figsize=(12, 6))
    missing_values[missing_values > 0].plot(kind='bar')
    plt.title('Missing Values by Feature')
    plt.xlabel('Features')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def correlation_analysis(dataset_train):
    """Perform correlation analysis and feature selection"""
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Split data FIRST to avoid data leakage
    train_temp, val_temp = train_test_split(dataset_train, test_size=0.2, random_state=42)
    
    # Only use training portion for feature selection
    numerical = train_temp.select_dtypes(include=['number'])
    corr = numerical.corr()
    
    # Get the top features most correlated with SalePrice
    top_features = corr['SalePrice'].sort_values(ascending=False)
    print("Top 20 features correlated with SalePrice:")
    print(top_features.head(20))
    
    # Select features with HIGH correlation only - AGGRESSIVE feature selection
    important_features = top_features[top_features > 0.25].index  # Much higher threshold
    print(f"\nFeatures with correlation > 0.25: {len(important_features)}")
    print(important_features.tolist())
    
    # Limit to top 15 features maximum to prevent overfitting
    if len(important_features) > 15:
        important_features = top_features[top_features > 0.25].head(15).index
        print(f"\nLimited to top 15 features to prevent overfitting:")
        print(important_features.tolist())
    
    # Visualize correlation heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", aspect='auto')
    plt.colorbar()
    plt.title("Correlation Heatmap")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.show()
    
    return important_features

def preprocess_data(dataset_train, dataset_test):
    """Improved data preprocessing"""
    print("\n=== DATA PREPROCESSING ===")
    
    # Combine train and test datasets for consistent processing
    data = pd.concat([dataset_train, dataset_test], sort=False)
    
    # Handle missing values more intelligently
    num_data = data.select_dtypes(include=["int64", "float64"])
    imputer = SimpleImputer(strategy="median")
    num_data_imputed = pd.DataFrame(imputer.fit_transform(num_data), columns=num_data.columns)
    
    # Handle categorical variables more carefully
    cat_data = data.select_dtypes(include=["object"])
    cat_data_imputed = cat_data.fillna('Unknown')
    
    # Encode categorical variables
    encoder = LabelEncoder()
    for col in cat_data_imputed.columns:
        cat_data_imputed[col] = encoder.fit_transform(cat_data_imputed[col])
    
    # Merge numerical and categorical data
    num_data_imputed = num_data_imputed.reset_index(drop=True)
    cat_data_imputed = cat_data_imputed.reset_index(drop=True)
    data_preprocessed = pd.concat([num_data_imputed, cat_data_imputed], axis=1)
    
    print(f"Preprocessed data shape: {data_preprocessed.shape}")
    return data_preprocessed

def prepare_modeling_data(data_preprocessed, dataset_train, dataset_test, important_features):
    """Prepare data for modeling"""
    print("\n=== PREPARING MODELING DATA ===")
    
    X = data_preprocessed[:len(dataset_train)]
    X_test = data_preprocessed[len(dataset_train):]
    
    # Select features and handle missing columns
    available_features = [f for f in important_features if f in X.columns]
    print(f"Available features: {len(available_features)}")
    
    X = X[available_features].drop('SalePrice', axis=1, errors='ignore')
    X_test = X_test[available_features].drop('SalePrice', axis=1, errors='ignore')
    y = dataset_train["SalePrice"]
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning to prevent overfitting"""
    print("\n=== HYPERPARAMETER TUNING ===")
    print("Starting hyperparameter tuning...")
    
    # Define parameter grid for hyperparameter tuning - AGGRESSIVE overfitting prevention
    param_grid = {
        'n_estimators': [50, 100],  # Fewer trees to reduce complexity
        'max_depth': [3, 4, 5, 6],  # Much more conservative depth
        'min_samples_split': [10, 15, 20],  # Very restrictive to reduce overfitting
        'min_samples_leaf': [5, 8, 10],    # Very restrictive to reduce overfitting
        'max_features': ['sqrt', 0.2, 0.25],  # Fewer features to reduce complexity
        'bootstrap': [True],
        'oob_score': [True],
        'max_samples': [0.5, 0.6, 0.7],  # More aggressive bagging
        'ccp_alpha': [0.0, 0.001, 0.01]  # Cost complexity pruning
    }
    
    # Create base model with early stopping and strong regularization
    base_model = RandomForestRegressor(
        random_state=42, 
        warm_start=True,
        max_leaf_nodes=50,  # Limit leaf nodes
        min_weight_fraction_leaf=0.1  # Additional regularization
    )
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (RMSE): {np.sqrt(-grid_search.best_score_):.2f}")
    
    # Use best model and check for overfitting
    best_model = grid_search.best_estimator_
    
    # Quick overfitting check
    train_score = best_model.score(X_train, y_train)
    cv_score = grid_search.best_score_
    print(f"\nOverfitting check:")
    print(f"Training R²: {train_score:.4f}")
    print(f"CV R²: {cv_score:.4f}")
    print(f"Gap: {train_score - cv_score:.4f} (smaller is better)")
    
    return best_model

def train_and_evaluate_model(best_model, X_train, X_val, y_train, y_val):
    """Train and evaluate the improved model"""
    print("\n=== MODEL TRAINING AND EVALUATION ===")
    
    # Train the improved model with best parameters
    best_model.fit(X_train, y_train)
    
    # Also train a Ridge model for ensemble
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    ridge_model = Ridge(alpha=10.0, random_state=42)  # Strong regularization
    ridge_model.fit(X_train_scaled, y_train)
    
    # Make predictions from both models
    y_train_pred_improved = best_model.predict(X_train)
    y_val_pred_improved = best_model.predict(X_val)
    
    y_train_pred_ridge = ridge_model.predict(X_train_scaled)
    y_val_pred_ridge = ridge_model.predict(X_val_scaled)
    
    # Ensemble predictions (average of both models)
    y_train_pred_ensemble = (y_train_pred_improved + y_train_pred_ridge) / 2
    y_val_pred_ensemble = (y_val_pred_improved + y_val_pred_ridge) / 2
    
    # Evaluate ensemble model performance
    train_mape_improved = mean_absolute_percentage_error(y_train, y_train_pred_ensemble)
    train_mse_improved = mean_squared_error(y_train, y_train_pred_ensemble)
    val_mape_improved = mean_absolute_percentage_error(y_val, y_val_pred_ensemble)
    val_mse_improved = mean_squared_error(y_val, y_val_pred_ensemble)
    
    # Print improved results
    print("=== ENSEMBLE MODEL RESULTS ===")
    print(f"Training MAPE: {train_mape_improved:.2%}, Training MSE: {train_mse_improved:.2f}")
    print(f"Validation MAPE: {val_mape_improved:.2%}, Validation MSE: {val_mse_improved:.2f}")
    print(f"Out-of-bag score: {best_model.oob_score_:.4f}")
    
    # Compare with original model
    print("\n=== COMPARISON WITH ORIGINAL MODEL ===")
    print(f"Original - Training MAPE: 7.99%, Validation MAPE: 12.14%")
    print(f"Improved - Training MAPE: {train_mape_improved:.2%}, Validation MAPE: {val_mape_improved:.2%}")
    
    # Check if overfitting is reduced
    overfitting_reduction = (val_mape_improved - train_mape_improved) - (12.14 - 7.99)
    print(f"Overfitting reduction: {overfitting_reduction:.2f} percentage points")
    
    return train_mape_improved, val_mape_improved

def analyze_feature_importance(best_model, X_train):
    """Analyze feature importance"""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 most important features:")
    print(feature_importance.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Feature Importance - Improved Model')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def cross_validation_evaluation(best_model, X, y):
    """Perform cross-validation evaluation"""
    print("\n=== CROSS-VALIDATION EVALUATION ===")
    print("Performing cross-validation...")
    
    # Use more folds and add overfitting detection
    from sklearn.model_selection import KFold
    
    # Create custom CV splits
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    improved_scores = cross_val_score(best_model, X, y, cv=cv, scoring='neg_mean_squared_error')
    improved_rmse_scores = np.sqrt(-improved_scores)
    
    # Check for overfitting in CV
    train_scores = []
    for train_idx, val_idx in cv.split(X):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        best_model.fit(X_fold_train, y_fold_train)
        train_score = best_model.score(X_fold_train, y_fold_train)
        val_score = best_model.score(X_fold_val, y_fold_val)
        train_scores.append(train_score)
    
    avg_train_score = np.mean(train_scores)
    avg_cv_score = np.mean(improved_scores)
    
    print("=== CROSS-VALIDATION RESULTS ===")
    print(f"Original model CV RMSE: 139.14")
    print(f"Improved model CV RMSE: {improved_rmse_scores.mean():.2f} (+/- {improved_rmse_scores.std() * 2:.2f})")
    print(f"\nOverfitting check in CV:")
    print(f"Average Training R²: {avg_train_score:.4f}")
    print(f"Average CV R²: {avg_cv_score:.4f}")
    print(f"CV Gap: {avg_train_score - avg_cv_score:.4f} (smaller is better)")
    
    return improved_rmse_scores

def generate_predictions(best_model, X_test, dataset_test, X_train, y_train):
    """Generate final predictions"""
    print("\n=== GENERATING PREDICTIONS ===")
    
    # Make final predictions with improved model
    final_predictions_rf = best_model.predict(X_test)
    
    # Also generate Ridge predictions for ensemble
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    final_predictions_ridge = ridge_model.predict(X_test_scaled)
    
    # Ensemble predictions
    final_predictions = (final_predictions_rf + final_predictions_ridge) / 2
    
    # Save improved submission file
    improved_submission = pd.DataFrame({
        "Id": dataset_test["Id"],
        "SalePrice": final_predictions
    })
    improved_submission.to_csv("improved_submission.csv", index=False)
    print("Improved submission file saved as 'improved_submission.csv'")
    
    # Display first few predictions
    print("\nFirst 10 predictions:")
    print(improved_submission.head(10))
    
    return improved_submission

def compare_with_ridge(X_train, X_val, y_train, y_val, improved_rmse_scores, val_mape_improved):
    """Compare with Ridge Regression"""
    print("\n=== RIDGE REGRESSION COMPARISON ===")
    print("Training Ridge Regression for comparison...")
    
    # Scale the features for Ridge regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train Ridge regression with cross-validation and hyperparameter tuning
    ridge_param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge_grid = GridSearchCV(Ridge(random_state=42), ridge_param_grid, cv=5, scoring='neg_mean_squared_error')
    ridge_grid.fit(X_train_scaled, y_train)
    ridge_model = ridge_grid.best_estimator_
    
    ridge_scores = cross_val_score(ridge_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    ridge_rmse = np.sqrt(-ridge_scores.mean())
    
    print("=== RIDGE REGRESSION COMPARISON ===")
    print(f"Ridge CV RMSE: {ridge_rmse:.2f}")
    print(f"Random Forest CV RMSE: {improved_rmse_scores.mean():.2f}")
    
    # Train Ridge on full training data
    ridge_model.fit(X_train_scaled, y_train)
    ridge_val_pred = ridge_model.predict(X_val_scaled)
    ridge_val_mape = mean_absolute_percentage_error(y_val, ridge_val_pred)
    
    print(f"\nRidge Validation MAPE: {ridge_val_mape:.2%}")
    print(f"Random Forest Validation MAPE: {val_mape_improved:.2%}")
    
    # Determine which model is better
    if ridge_val_mape < val_mape_improved:
        print("\nRidge Regression performs better on validation set!")
        best_final_model = ridge_model
        best_final_model_name = "Ridge Regression"
    else:
        print("\nRandom Forest performs better on validation set!")
        best_final_model = "Random Forest"  # Just store the name for now
        best_final_model_name = "Random Forest"
    
    print(f"\nBest model: {best_final_model_name}")
    return best_final_model_name, ridge_val_mape

def final_summary(best_final_model_name, best_model, feature_importance, X_train, X_val, X_test, 
                 train_mape_improved, val_mape_improved, ridge_val_mape=None):
    """Provide final model summary"""
    print("\n=== FINAL MODEL SUMMARY ===")
    print(f"Best model: {best_final_model_name}")
    
    if best_final_model_name == "Random Forest":
        print(f"Random Forest parameters: {best_model.get_params()}")
        print(f"Feature importance top 5: {feature_importance.head(5)['feature'].tolist()}")
    else:
        print(f"Ridge regression typically has less overfitting due to regularization")
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features used: {X_train.shape[1]}")
    
    print("\n=== OVERFITTING ANALYSIS ===")
    print(f"Original model overfitting gap: 12.14% - 7.99% = 4.15 percentage points")
    
    if best_final_model_name == "Random Forest":
        current_overfitting_gap = val_mape_improved - train_mape_improved
        print(f"Improved model overfitting gap: {val_mape_improved:.2%} - {train_mape_improved:.2%} = {current_overfitting_gap:.2f} percentage points")
        improvement = 4.15 - current_overfitting_gap
        print(f"Overfitting reduction: {improvement:.2f} percentage points")
    else:
        print("Ridge regression typically has less overfitting due to regularization")
        if ridge_val_mape:
            print(f"Ridge validation MAPE: {ridge_val_mape:.2%}")
    
    print("\nModel training and evaluation complete!")
    print("Check 'improved_submission.csv' for your predictions.")

def main():
    """Main function to run the complete pipeline"""
    print("🏠 IMPROVED HOUSING PRICES PREDICTION MODEL")
    print("=" * 50)
    
    # Step 1: Load data
    dataset_train, dataset_test = load_data()
    if dataset_train is None:
        return
    
    # Step 2: Analyze missing values
    analyze_missing_values(dataset_train)
    
    # Step 3: Correlation analysis and feature selection
    important_features = correlation_analysis(dataset_train)
    
    # Step 4: Preprocess data
    data_preprocessed = preprocess_data(dataset_train, dataset_test)
    
    # Step 5: Prepare modeling data
    X_train, X_val, X_test, y_train, y_val, y = prepare_modeling_data(
        data_preprocessed, dataset_train, dataset_test, important_features
    )
    
    # Step 6: Hyperparameter tuning
    best_model = hyperparameter_tuning(X_train, y_train)
    
    # Step 7: Train and evaluate model
    train_mape_improved, val_mape_improved = train_and_evaluate_model(
        best_model, X_train, X_val, y_train, y_val
    )
    
    # Step 8: Analyze feature importance
    feature_importance = analyze_feature_importance(best_model, X_train)
    
    # Step 9: Cross-validation evaluation
    improved_rmse_scores = cross_validation_evaluation(best_model, X_train, y_train)
    
    # Step 10: Generate predictions
    improved_submission = generate_predictions(best_model, X_test, dataset_test, X_train, y_train)
    
    # Step 11: Compare with Ridge regression
    best_final_model_name, ridge_val_mape = compare_with_ridge(
        X_train, X_val, y_train, y_val, improved_rmse_scores, val_mape_improved
    )
    
    # Step 12: Final summary
    final_summary(best_final_model_name, best_model, feature_importance, 
                 X_train, X_val, X_test, train_mape_improved, val_mape_improved, ridge_val_mape)
    
    print("\n🎉 All done! Your improved model is ready!")

if __name__ == "__main__":
    main()
