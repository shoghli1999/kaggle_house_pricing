#!/usr/bin/env python3
"""
Housing Prices Prediction - Interactive Web Dashboard
A Streamlit application to visualize and interact with the housing prices analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add the current directory to path to import our housing prices module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our housing prices functions
try:
    from housing_prices_improved import (
        load_data, analyze_missing_values, correlation_analysis,
        preprocess_data, prepare_modeling_data, hyperparameter_tuning,
        train_and_evaluate_model, analyze_feature_importance,
        cross_validation_evaluation, generate_predictions,
        compare_with_ridge, comprehensive_overfitting_check
    )
except ImportError:
    st.error("Could not import housing_prices_improved module. Make sure it's in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Housing Prices Analysis Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Housing Prices Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Overview", "🔍 Data Analysis", "🏗️ Model Training", "📈 Results & Evaluation", "🎯 Predictions", "⚙️ Settings"]
    )
    
    if page == "📊 Overview":
        show_overview()
    elif page == "🔍 Data Analysis":
        show_data_analysis()
    elif page == "🏗️ Model Training":
        show_model_training()
    elif page == "📈 Results & Evaluation":
        show_results_evaluation()
    elif page == "🎯 Predictions":
        show_predictions()
    elif page == "⚙️ Settings":
        show_settings()

def show_overview():
    """Show project overview and summary"""
    st.header("📊 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## About This Project
        
        This is an **improved housing prices prediction model** that addresses overfitting issues 
        in Random Forest models. The project focuses on:
        
        - **Aggressive feature selection** (correlation > 0.25, max 15 features)
        - **Conservative hyperparameters** to prevent overfitting
        - **Regularization techniques** (cost complexity pruning, leaf node limits)
        - **Cross-validation** for robust evaluation
        - **Ensemble approach** with Ridge regression
        - **Comprehensive overfitting detection** and analysis
        
        ## Key Features
        
        ✅ **Data Leakage Prevention**: Strict train/validation separation  
        ✅ **Feature Selection**: Only high-correlation features  
        ✅ **Hyperparameter Tuning**: GridSearchCV with cross-validation  
        ✅ **Overfitting Detection**: Multiple validation metrics  
        ✅ **Model Comparison**: Random Forest vs Ridge Regression  
        ✅ **Ensemble Predictions**: Combined model outputs  
        """)
    
    with col2:
        st.markdown("""
        ## Quick Stats
        
        **Dataset**: Kaggle Housing Prices  
        **Target**: SalePrice  
        **Algorithm**: Random Forest + Ridge  
        **Validation**: 5-fold Cross-validation  
        **Features**: Max 15 (correlation > 0.25)  
        """)
        
        # Check if data files exist
        if os.path.exists("train.csv") and os.path.exists("test.csv"):
            st.success("✅ Data files found")
        else:
            st.error("❌ Data files not found")
            st.info("Please ensure 'train.csv' and 'test.csv' are in the same directory")
        
        if os.path.exists("improved_submission.csv"):
            st.success("✅ Model predictions available")
        else:
            st.info("ℹ️ No predictions generated yet")

def show_data_analysis():
    """Show data analysis and exploration"""
    st.header("🔍 Data Analysis")
    
    # Load data
    with st.spinner("Loading datasets..."):
        dataset_train, dataset_test = load_data()
    
    if dataset_train is None:
        st.error("Could not load data. Please check your data files.")
        return
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Info", "❌ Missing Values", "🔗 Correlations", "📊 Feature Distribution"])
    
    with tab1:
        st.subheader("Dataset Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Training Data:**")
            st.write(f"Shape: {dataset_train.shape}")
            st.write(f"Columns: {dataset_train.columns.tolist()}")
            
        with col2:
            st.write("**Test Data:**")
            st.write(f"Shape: {dataset_test.shape}")
            st.write(f"Columns: {dataset_test.columns.tolist()}")
        
        # Show first few rows
        st.subheader("First 5 rows of training data")
        st.dataframe(dataset_train.head())
        
        # Data types
        st.subheader("Data Types")
        st.write(dataset_train.dtypes.value_counts())
    
    with tab2:
        st.subheader("Missing Values Analysis")
        
        # Calculate missing values
        missing_values = dataset_train.isnull().sum().sort_values(ascending=False)
        missing_values = missing_values[missing_values > 0]
        
        if len(missing_values) > 0:
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            missing_values.plot(kind='bar', ax=ax)
            plt.title('Missing Values by Feature')
            plt.xlabel('Features')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show missing values table
            st.write("**Features with missing values:**")
            st.dataframe(missing_values.reset_index().rename(columns={'index': 'Feature', 0: 'Missing Count'}))
        else:
            st.success("✅ No missing values found in the dataset!")
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Perform correlation analysis
        with st.spinner("Analyzing correlations..."):
            important_features = correlation_analysis(dataset_train)
        
        if important_features is not None:
            st.success(f"✅ Selected {len(important_features)} important features")
            st.write("**Selected features:**")
            st.write(list(important_features))
            
            # Show correlation heatmap
            numerical = dataset_train.select_dtypes(include=['number'])
            corr = numerical.corr()
            
            # Create interactive heatmap with plotly
            fig = px.imshow(
                corr,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Distribution")
        
        # Select numerical features for distribution plots
        numerical_cols = dataset_train.select_dtypes(include=['number']).columns.tolist()
        selected_features = st.multiselect(
            "Select features to visualize:",
            numerical_cols,
            default=numerical_cols[:5] if len(numerical_cols) >= 5 else numerical_cols
        )
        
        if selected_features:
            # Create distribution plots
            fig = make_subplots(
                rows=len(selected_features), cols=1,
                subplot_titles=selected_features,
                vertical_spacing=0.1
            )
            
            for i, feature in enumerate(selected_features, 1):
                fig.add_trace(
                    go.Histogram(x=dataset_train[feature], name=feature),
                    row=i, col=1
                )
            
            fig.update_layout(height=200*len(selected_features), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    """Show model training process"""
    st.header("🏗️ Model Training")
    
    # Check if data is available
    if not (os.path.exists("train.csv") and os.path.exists("test.csv")):
        st.error("Please load data first in the Data Analysis tab.")
        return
    
    # Load data
    with st.spinner("Loading datasets..."):
        dataset_train, dataset_test = load_data()
    
    if dataset_train is None:
        st.error("Could not load data.")
        return
    
    # Create tabs for different training aspects
    tab1, tab2, tab3 = st.tabs(["🔧 Preprocessing", "🎯 Feature Selection", "🤖 Model Training"])
    
    with tab1:
        st.subheader("Data Preprocessing")
        
        if st.button("Run Data Preprocessing"):
            with st.spinner("Preprocessing data..."):
                data_preprocessed = preprocess_data(dataset_train, dataset_test)
                st.success(f"✅ Preprocessing complete! Shape: {data_preprocessed.shape}")
                
                # Store in session state for later use
                st.session_state['data_preprocessed'] = data_preprocessed
    
    with tab2:
        st.subheader("Feature Selection")
        
        if st.button("Run Feature Selection"):
            with st.spinner("Selecting features..."):
                important_features = correlation_analysis(dataset_train)
                if important_features is not None:
                    st.session_state['important_features'] = important_features
                    st.success(f"✅ Selected {len(important_features)} features")
                    
                    # Show selected features
                    st.write("**Selected features:**")
                    for i, feature in enumerate(important_features, 1):
                        st.write(f"{i}. {feature}")
    
    with tab3:
        st.subheader("Model Training")
        
        # Check if preprocessing and feature selection are done
        if 'data_preprocessed' not in st.session_state:
            st.warning("Please run preprocessing first.")
            return
        
        if 'important_features' not in st.session_state:
            st.warning("Please run feature selection first.")
            return
        
        if st.button("Start Model Training"):
            with st.spinner("Training model..."):
                try:
                    # Prepare modeling data
                    X_train, X_val, X_test, y_train, y_val, y = prepare_modeling_data(
                        st.session_state['data_preprocessed'], 
                        dataset_train, 
                        dataset_test, 
                        st.session_state['important_features']
                    )
                    
                    st.session_state['X_train'] = X_train
                    st.session_state['X_val'] = X_val
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_val'] = y_val
                    st.session_state['y'] = y
                    
                    st.success("✅ Data preparation complete!")
                    
                    # Hyperparameter tuning
                    st.info("Running hyperparameter tuning...")
                    best_model = hyperparameter_tuning(X_train, y_train)
                    st.session_state['best_model'] = best_model
                    
                    st.success("✅ Hyperparameter tuning complete!")
                    
                    # Train and evaluate
                    st.info("Training and evaluating model...")
                    train_mape, val_mape = train_and_evaluate_model(
                        best_model, X_train, X_val, y_train, y_val
                    )
                    
                    st.session_state['train_mape'] = train_mape
                    st.session_state['val_mape'] = val_mape
                    
                    st.success("✅ Model training complete!")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")

def show_results_evaluation():
    """Show model results and evaluation"""
    st.header("📈 Results & Evaluation")
    
    # Check if model is trained
    if 'best_model' not in st.session_state:
        st.warning("Please train the model first in the Model Training tab.")
        return
    
    # Create tabs for different evaluation aspects
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "🎯 Feature Importance", "🔄 Cross-Validation", "⚠️ Overfitting Analysis"])
    
    with tab1:
        st.subheader("Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training MAPE", f"{st.session_state.get('train_mape', 0):.2%}")
        with col2:
            st.metric("Validation MAPE", f"{st.session_state.get('val_mape', 0):.2%}")
        
        # Performance comparison
        if 'train_mape' in st.session_state and 'val_mape' in st.session_state:
            train_mape = st.session_state['train_mape']
            val_mape = st.session_state['val_mape']
            
            # Calculate overfitting gap
            overfitting_gap = val_mape - train_mape
            
            st.subheader("Overfitting Analysis")
            if overfitting_gap < 0.05:
                st.success(f"✅ Good: Overfitting gap is acceptable ({overfitting_gap:.2%})")
            else:
                st.warning(f"⚠️ Warning: Overfitting gap is high ({overfitting_gap:.2%})")
    
    with tab2:
        st.subheader("Feature Importance")
        
        if st.button("Analyze Feature Importance"):
            with st.spinner("Analyzing feature importance..."):
                try:
                    feature_importance = analyze_feature_importance(
                        st.session_state['best_model'], 
                        st.session_state['X_train']
                    )
                    
                    # Create interactive bar chart
                    fig = px.bar(
                        feature_importance.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 15 Feature Importance Scores"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table
                    st.write("**Feature importance scores:**")
                    st.dataframe(feature_importance.head(15))
                    
                except Exception as e:
                    st.error(f"Error analyzing feature importance: {str(e)}")
    
    with tab3:
        st.subheader("Cross-Validation Results")
        
        if st.button("Run Cross-Validation"):
            with st.spinner("Running cross-validation..."):
                try:
                    cv_scores = cross_validation_evaluation(
                        st.session_state['best_model'],
                        st.session_state['X_train'],
                        st.session_state['y_train']
                    )
                    
                    # Display CV results
                    st.write(f"**Cross-validation RMSE:** {cv_scores.mean():.2f} ± {cv_scores.std() * 2:.2f}")
                    
                    # Create CV scores distribution
                    fig = px.histogram(
                        x=cv_scores,
                        title="Cross-Validation RMSE Distribution",
                        labels={'x': 'RMSE', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during cross-validation: {str(e)}")
    
    with tab4:
        st.subheader("Comprehensive Overfitting Check")
        
        if st.button("Run Overfitting Analysis"):
            with st.spinner("Analyzing overfitting..."):
                try:
                    feature_importance = analyze_feature_importance(
                        st.session_state['best_model'], 
                        st.session_state['X_train']
                    )
                    
                    overfitting_score = comprehensive_overfitting_check(
                        st.session_state['best_model'],
                        st.session_state['X_train'],
                        st.session_state['X_val'],
                        st.session_state['y_train'],
                        st.session_state['y_val'],
                        feature_importance
                    )
                    
                    st.success(f"✅ Overfitting analysis complete! Score: {overfitting_score}/5")
                    
                except Exception as e:
                    st.error(f"Error during overfitting analysis: {str(e)}")

def show_predictions():
    """Show model predictions"""
    st.header("🎯 Model Predictions")
    
    # Check if model is trained
    if 'best_model' not in st.session_state:
        st.warning("Please train the model first in the Model Training tab.")
        return
    
    # Check if test data is available
    if 'X_test' not in st.session_state:
        st.warning("Test data not available. Please train the model first.")
        return
    
    # Generate predictions
    if st.button("Generate Predictions"):
        with st.spinner("Generating predictions..."):
            try:
                # Load test dataset for IDs
                dataset_train, dataset_test = load_data()
                
                predictions = generate_predictions(
                    st.session_state['best_model'],
                    st.session_state['X_test'],
                    dataset_test,
                    st.session_state['X_train'],
                    st.session_state['y_train']
                )
                
                st.success("✅ Predictions generated successfully!")
                
                # Show predictions
                st.subheader("Sample Predictions")
                st.dataframe(predictions.head(20))
                
                # Download predictions
                csv = predictions.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="housing_predictions.csv",
                    mime="text/csv"
                )
                
                # Show prediction statistics
                st.subheader("Prediction Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Min Price", f"${predictions['SalePrice'].min():,.0f}")
                with col2:
                    st.metric("Max Price", f"${predictions['SalePrice'].max():,.0f}")
                with col3:
                    st.metric("Mean Price", f"${predictions['SalePrice'].mean():,.0f}")
                
                # Create prediction distribution
                fig = px.histogram(
                    predictions,
                    x='SalePrice',
                    title="Predicted House Price Distribution",
                    labels={'SalePrice': 'Predicted Price ($)', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")

def show_settings():
    """Show application settings and configuration"""
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("Data Files")
    
    # Check data files
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists("train.csv"):
            st.success("✅ train.csv found")
            train_size = os.path.getsize("train.csv") / (1024 * 1024)  # MB
            st.write(f"Size: {train_size:.2f} MB")
        else:
            st.error("❌ train.csv not found")
    
    with col2:
        if os.path.exists("test.csv"):
            st.success("✅ test.csv found")
            test_size = os.path.getsize("test.csv") / (1024 * 1024)  # MB
            st.write(f"Size: {test_size:.2f} MB")
        else:
            st.error("❌ test.csv not found")
    
    st.subheader("Model Status")
    
    if 'best_model' in st.session_state:
        st.success("✅ Model trained and ready")
        st.write(f"Model type: {type(st.session_state['best_model']).__name__}")
    else:
        st.info("ℹ️ No model trained yet")
    
    st.subheader("Session State")
    
    # Show all session state variables
    if st.button("Show Session State"):
        st.write("**Current session state variables:**")
        for key, value in st.session_state.items():
            if hasattr(value, 'shape'):
                st.write(f"{key}: {value.shape}")
            else:
                st.write(f"{key}: {type(value).__name__}")
    
    # Clear session state
    if st.button("Clear Session State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Session state cleared")

if __name__ == "__main__":
    main()
