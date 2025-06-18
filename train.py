"""
Git-Surgeon Intent Classifier Training Script

This script trains a high-performance NLP model to classify git command intents
from natural language queries. It implements advanced techniques including:

- Enhanced text preprocessing and feature engineering
- Hyperparameter optimization
- Cross-validation
- Model ensembling or best model selection
- Rich evaluation metrics
- Confusion matrix visualization
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from typing import Dict, Any, Tuple, List, Optional
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV  # Add calibration for SVM
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Preprocess the dataset for training.
    
    Args:
        df: Input DataFrame with text and intent columns
        
    Returns:
        Processed DataFrame and label encoder
    """
    logger.info("Preprocessing data...")
    
    # Verify required columns exist
    if 'text' not in df.columns or 'intent' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'intent' columns")
    
    # Clean text data
    df['text'] = df['text'].str.lower()
    
    # Add feature engineering
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['has_number'] = df['text'].str.contains(r'\d').astype(int)
    
    # Add command-specific features
    df['has_squash_keyword'] = df['text'].str.contains(r'squash|combin|merg|collaps').astype(int)
    df['has_reword_keyword'] = df['text'].str.contains(r'reword|messag|edit|chang').astype(int)
    df['has_branch_keyword'] = df['text'].str.contains(r'branch|name|renam').astype(int)
    df['has_rebase_keyword'] = df['text'].str.contains(r'rebas|onto|top of').astype(int)

    # Encode labels
    label_encoder = LabelEncoder()
    df['intent_encoded'] = label_encoder.fit_transform(df['intent'])
    
    logger.info(f"Data preprocessed: {df.shape[0]} samples, {df.shape[1]} features")
    return df, label_encoder

def create_model_pipeline() -> Dict[str, Pipeline]:
    """
    Create different model pipelines for comparison.
    
    Returns:
        Dictionary of named model pipelines
    """
    logger.info("Creating model pipelines...")
    
    # Text feature processor
    text_features = Pipeline([
        ('vectorizer', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=2,
            use_idf=True,
            sublinear_tf=True
        ))
    ])
    
    # Build pipelines for different models
    pipelines = {
        'logistic': Pipeline([
            ('features', ColumnTransformer([
                ('text', text_features, 'text'),
            ], remainder='passthrough')),
            ('classifier', LogisticRegression(
                C=10.0,
                max_iter=1000,
                class_weight='balanced',
                solver='newton-cg',  # Modern solver that works well for multiclass
                # Removed multi_class parameter to avoid deprecation warning
            ))
        ]),
        
        'svm': Pipeline([
            ('features', ColumnTransformer([
                ('text', text_features, 'text'),
            ], remainder='passthrough')),
            ('classifier', CalibratedClassifierCV(
                LinearSVC(
                    C=1.0,
                    class_weight='balanced',
                    dual=False,
                    max_iter=2000
                ),
                cv=3,  # Use 3-fold CV for calibration
                method='sigmoid'  # Platt scaling
            ))
        ]),
        
        'forest': Pipeline([
            ('features', ColumnTransformer([
                ('text', Pipeline([
                    ('vectorizer', CountVectorizer(ngram_range=(1, 2), max_features=5000))
                ]), 'text'),
            ], remainder='passthrough')),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced'
            ))
        ])
    }
    
    return pipelines

def optimize_hyperparameters(X_train: pd.DataFrame, y_train: np.ndarray, 
                            pipelines: Dict[str, Pipeline], cv: int = 3) -> Dict[str, BaseEstimator]:
    """
    Optimize hyperparameters for each model pipeline.
    
    Args:
        X_train: Training features
        y_train: Training labels
        pipelines: Model pipelines to optimize
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary of optimized models
    """
    logger.info("Optimizing hyperparameters...")
    
    best_models = {}
    
    # Define parameter grids for each model
    param_grids = {
        'logistic': {
            'classifier__C': [0.1, 1.0, 10.0],
            'features__text__vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'features__text__vectorizer__max_features': [5000, 10000]
        },
        'svm': {
            # Fixed: properly nested parameter for the inner estimator
            'classifier__estimator__C': [0.1, 1.0, 5.0],
            'features__text__vectorizer__ngram_range': [(1, 2), (1, 3)],
            'features__text__vectorizer__max_features': [5000, 10000]
        },
        'forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 30, 50],
            'features__text__vectorizer__max_features': [3000, 5000]
        }
    }
    
    # For each model, perform grid search
    for name, pipeline in pipelines.items():
        logger.info(f"Optimizing {name} model...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[name],
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        logger.info(f"Best CV score for {name}: {grid_search.best_score_:.4f}")
        
        best_models[name] = grid_search.best_estimator_
    
    return best_models

def create_ensemble(models: Dict[str, BaseEstimator]) -> VotingClassifier:
    """
    Create an ensemble model from the optimized models.
    
    Args:
        models: Dictionary of optimized models
        
    Returns:
        Voting classifier ensemble
    """
    logger.info("Creating ensemble model...")
    
    # Ensure all models have predict_proba method
    valid_estimators = []
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            valid_estimators.append((name, model))
        else:
            logger.warning(f"Model {name} does not support probability prediction, excluding from ensemble")
    
    if len(valid_estimators) < 2:
        logger.warning("Not enough models support probability prediction for soft voting. Using hard voting instead.")
        voting = 'hard'
    else:
        voting = 'soft'
    
    ensemble = VotingClassifier(estimators=valid_estimators, voting=voting)
    
    return ensemble

def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: np.ndarray, 
                  label_encoder: LabelEncoder, output_dir: str) -> Dict[str, Any]:
    """
    Evaluate model performance and generate visualization.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: True labels
        label_encoder: Label encoder for intent names
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of performance metrics
    """
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Generate classification report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save confusion matrix
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")
    
    # If model has predict_proba, calculate confidence
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_test)
        confidences = np.max(probas, axis=1)
        avg_confidence = np.mean(confidences)
        logger.info(f"Average prediction confidence: {avg_confidence:.4f}")
    else:
        avg_confidence = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_confidence': avg_confidence,
        'report': report
    }

def train_model(data_path: str, model_path: str, output_dir: str,
                test_size: float = 0.2, random_state: int = 42,
                optimize: bool = True, ensemble: bool = True,
                single_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Train and save a high-performance intent classification model.
    
    Args:
        data_path: Path to the TSV data file
        model_path: Path to save the trained model
        output_dir: Directory to save outputs and visualizations
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        optimize: Whether to run hyperparameter optimization
        ensemble: Whether to create an ensemble model
        single_model: If specified, train only this model type ('logistic', 'svm', or 'forest')
        
    Returns:
        Dictionary with training results and metrics
    """
    start_time = datetime.now()
    logger.info(f"Starting model training at {start_time}")
    
    # Create output directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path, sep="\t", names=["text", "intent"])
    logger.info(f"Dataset loaded: {len(df)} samples")
    
    # Display class distribution
    class_dist = df['intent'].value_counts()
    logger.info("Class distribution:")
    for label, count in class_dist.items():
        logger.info(f"  {label}: {count} samples ({count/len(df):.1%})")
    
    # Preprocess data
    df, label_encoder = preprocess_data(df)
    
    # Split into features and target
    X = df.drop(['intent', 'intent_encoded'], axis=1)
    y = df['intent_encoded']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Create model pipelines
    pipelines = create_model_pipeline()
    
    # If single_model is specified, only use that model
    if single_model:
        if single_model not in pipelines:
            logger.warning(f"Requested model '{single_model}' not found. Using all models.")
        else:
            logger.info(f"Training only the {single_model} model as requested.")
            pipelines = {single_model: pipelines[single_model]}
    
    # Training process
    if optimize:
        # Optimize hyperparameters
        best_models = optimize_hyperparameters(X_train, y_train, pipelines)
        
        # If single model is specified, no need for ensemble
        if single_model and len(best_models) == 1:
            final_model = next(iter(best_models.values()))
        elif ensemble and len(best_models) > 1:
            # Create and train ensemble
            logger.info("Training ensemble model...")
            # Only use models that have predict_proba
            final_model = create_ensemble(best_models)
            final_model.fit(X_train, y_train)
        else:
            # Select best individual model
            logger.info("Selecting best individual model...")
            model_scores = {}
            for name, model in best_models.items():
                cv_score = cross_val_score(model, X_train, y_train, 
                                         cv=5, scoring='f1_weighted').mean()
                model_scores[name] = cv_score
                logger.info(f"CV score for {name}: {cv_score:.4f}")
            
            best_model_name = max(model_scores, key=model_scores.get)
            logger.info(f"Best model: {best_model_name} with score {model_scores[best_model_name]:.4f}")
            final_model = best_models[best_model_name]
    else:
        # Use logistic regression with default parameters
        if single_model and single_model in pipelines:
            logger.info(f"Training {single_model} model with default parameters...")
            final_model = pipelines[single_model]
        else:
            logger.info("Training logistic regression model with default parameters...")
            final_model = pipelines['logistic']
        final_model.fit(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(final_model, X_test, y_test, label_encoder, output_dir)
    
    # Save the model
    logger.info(f"Saving model to {model_path}...")
    model_info = {
        'model': final_model,
        'label_encoder': label_encoder,
        'metrics': metrics,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'class_names': label_encoder.classes_.tolist()
    }
    joblib.dump(model_info, model_path)
    
    # Calculate total training time
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save training report
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Git-Surgeon Intent Classifier Training Report\n")
        f.write(f"=========================================\n\n")
        f.write(f"Training date: {model_info['training_date']}\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n")
        f.write(f"Number of intent classes: {len(label_encoder.classes_)}\n\n")
        f.write(f"Model type: {'Ensemble' if ensemble else 'Single'}\n")
        f.write(f"Optimization performed: {'Yes' if optimize else 'No'}\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"------------------\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        if metrics['avg_confidence']:
            f.write(f"Average confidence: {metrics['avg_confidence']:.4f}\n\n")
        f.write(f"Classification Report:\n")
        f.write(f"{metrics['report']}\n")
    
    logger.info(f"Saved training report to {report_path}")
    
    return {
        'model': final_model,
        'metrics': metrics,
        'training_time': training_time,
        'label_encoder': label_encoder
    }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Train Git-Surgeon Intent Classifier")
    parser.add_argument("--data", default="data/intents.tsv", help="Path to training data TSV file")
    parser.add_argument("--model", default="models/intent_classifier.pkl", help="Path to save the trained model")
    parser.add_argument("--output", default="models/training_outputs", help="Directory for training outputs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to reserve for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    parser.add_argument("--no-ensemble", action="store_true", help="Skip ensemble model creation")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode with minimal hyperparameter search")
    parser.add_argument("--single", choices=["logistic", "svm", "forest"], 
                       help="Train only a specific model type instead of an ensemble")
    
    args = parser.parse_args()
    
    # If quick mode is enabled, use a simpler approach
    if args.quick:
        logger.info("Running in quick mode with simplified training")
        # Create and train a simple logistic regression model
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('classifier', LogisticRegression(C=1.0, max_iter=1000))
        ])
        
        # Load data
        df = pd.read_csv(args.data, sep="\t", names=["text", "intent"])
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["intent"], test_size=args.test_size, random_state=args.seed)
        
        # Train and save
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)
        logger.info(f"Quick mode accuracy: {accuracy:.4f}")
        
        # Save the model
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        joblib.dump(pipeline, args.model)
        logger.info(f"Model saved to {args.model}")
    else:
        # Run full training
        train_model(
            data_path=args.data,
            model_path=args.model,
            output_dir=args.output,
            test_size=args.test_size,
            random_state=args.seed,
            optimize=not args.no_optimize,
            ensemble=not args.no_ensemble,
            single_model=args.single
        )

if __name__ == "__main__":
    main()
