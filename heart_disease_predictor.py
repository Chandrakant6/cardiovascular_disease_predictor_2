import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
import os
from joblib import dump, load
import time

warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    def __init__(self):
        self.models_dir = 'saved_models'
        self.viz_dir = 'visualizations'
        
        # Create necessary directories
        for directory in [self.models_dir, self.viz_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize all models with simplified parameters
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=25,  # Further reduced
                max_depth=4,      # Further reduced
                n_jobs=1,         # Disable parallel processing
                random_state=42
            ),
            "Linear SVM": Pipeline([
                ('scaler', StandardScaler()),
                ('svm', LinearSVC(
                    max_iter=500,  # Further reduced
                    random_state=42
                ))
            ]),
            "KNN": KNeighborsClassifier(
                n_neighbors=3,
                n_jobs=1          # Disable parallel processing
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=4,      # Further reduced
                random_state=42
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=500,     # Further reduced
                n_jobs=1,         # Disable parallel processing
                random_state=42
            )
        }
        
        self.scaler = RobustScaler()
        self.best_model = None
        self.model_results = {}
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the heart disease dataset"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(file_path, sep=';')
        
        # Sample the data to reduce size (take 10% of the data)
        df = df.sample(frac=0.1, random_state=42)
        print(f"Using {len(df)} samples for training")
        
        # Basic preprocessing
        df['age'] = (df['age'] / 365).astype(int)
        df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
        
        # Handle outliers
        numerical_cols = ['ap_hi', 'ap_lo', 'height', 'weight', 'bmi']
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        
        # Scale numerical features
        numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def analyze_data(self, df):
        """Perform basic data analysis"""
        print("\nPerforming data analysis...")
        
        # Basic statistics
        print("\nDataset Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Number of features: {len(df.columns)}")
        print(f"Target distribution:\n{df['cardio'].value_counts(normalize=True)}")
        
        # Create visualizations
        self.create_visualizations(df)
        
    def create_visualizations(self, df):
        """Create essential visualizations"""
        print("\nCreating visualizations...")
        plt.style.use('default')
        
        # 1. Correlation Matrix
        plt.figure(figsize=(10, 8))
        numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'cardio']
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'correlation_matrix.png'))
        plt.close()
        
        # 2. Distribution Plots
        plt.figure(figsize=(15, 5))
        
        # Age Distribution
        plt.subplot(1, 3, 1)
        sns.histplot(data=df, x='age', hue='cardio', bins=30)
        plt.title('Age Distribution')
        plt.xlabel('Age (years)')
        
        # Blood Pressure Distribution
        plt.subplot(1, 3, 2)
        sns.boxplot(data=df, x='cardio', y='ap_hi')
        plt.title('Systolic BP by Heart Disease')
        plt.xlabel('Heart Disease')
        plt.ylabel('Systolic BP')
        
        # BMI Distribution
        plt.subplot(1, 3, 3)
        sns.boxplot(data=df, x='cardio', y='bmi')
        plt.title('BMI by Heart Disease')
        plt.xlabel('Heart Disease')
        plt.ylabel('BMI')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'key_distributions.png'))
        plt.close()
    
    def train_single_model(self, name, model, X_train, X_test, y_train, y_test):
        """Train and evaluate a single model"""
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Use fewer CV folds for faster evaluation
        cv_scores = cross_val_score(model, X_train, y_train, cv=2, n_jobs=1)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        training_time = time.time() - start_time
        
        # Store results
        results = {
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'model': model,
            'training_time': training_time
        }
        
        print(f"\n{name} Results:")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.viz_dir, f'confusion_matrix_{name.lower().replace(" ", "_")}.png'))
        plt.close()
        
        return name, results
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate models sequentially"""
        print("\nTraining and evaluating models...")
        
        results = {}
        best_accuracy = 0
        
        # Train models sequentially instead of in parallel
        for name, model in self.models.items():
            name, result = self.train_single_model(name, model, X_train, X_test, y_train, y_test)
            results[name] = result
            
            # Update best model if needed
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                self.best_model = result['model']
                self.save_model(result['model'], name)
        
        self.model_results = results
        self.plot_model_comparison()
        return results
    
    def plot_model_comparison(self):
        """Create comparison plot for all models"""
        # Prepare data for plotting
        model_names = list(self.model_results.keys())
        accuracies = [results['accuracy'] for results in self.model_results.values()]
        cv_means = [results['cv_scores'].mean() for results in self.model_results.values()]
        cv_stds = [results['cv_scores'].std() * 2 for results in self.model_results.values()]
        training_times = [results['training_time'] for results in self.model_results.values()]
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy comparison
        plt.subplot(1, 3, 1)
        bars = plt.bar(model_names, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add accuracy values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Plot CV scores comparison
        plt.subplot(1, 3, 2)
        plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        plt.title('Cross-Validation Scores Comparison')
        plt.xlabel('Models')
        plt.ylabel('CV Score')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Plot training times
        plt.subplot(1, 3, 3)
        plt.bar(model_names, training_times)
        plt.title('Training Time Comparison')
        plt.xlabel('Models')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'model_comparison.png'))
        plt.close()
        
        # Print final comparison
        print("\nFinal Model Comparison:")
        print("\nModel\t\tAccuracy\tCV Score (mean ± std)\tTraining Time (s)")
        print("-" * 70)
        for name in model_names:
            results = self.model_results[name]
            print(f"{name:<20} {results['accuracy']:.4f}\t\t{results['cv_scores'].mean():.4f} ± {results['cv_scores'].std()*2:.4f}\t\t{results['training_time']:.2f}")
    
    def save_model(self, model, model_name):
        """Save the model and scaler using joblib"""
        model_path = os.path.join(self.models_dir, f"{model_name.lower().replace(' ', '_')}.joblib")
        dump(model, model_path, compress=3)
        
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        dump(self.scaler, scaler_path, compress=3)
        
        print(f"Model and scaler saved to {self.models_dir}")
    
    def load_model(self, model_name):
        """Load the saved model and scaler"""
        model_path = os.path.join(self.models_dir, f"{model_name.lower().replace(' ', '_')}.joblib")
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler files not found. Please train the model first.")
        
        self.best_model = load(model_path)
        self.scaler = load(scaler_path)
        print(f"{model_name} model and scaler loaded successfully")
    
    def predict(self, features):
        """Make predictions for new cases"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)
        probability = self.best_model.predict_proba(features_scaled)
        
        return prediction, probability

def main():
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Load and preprocess data
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cardio_train.csv")
    df = predictor.load_and_preprocess_data(file_path)
    
    # Analyze data
    predictor.analyze_data(df)
    
    # Prepare data for training
    X = df.drop(columns=['cardio'])
    y = df['cardio']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Train and evaluate models
    results = predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Print final results
    print("\nFinal Model Results:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"CV Score: {result['cv_scores'].mean():.4f} (+/- {result['cv_scores'].std() * 2:.4f})")

if __name__ == "__main__":
    main()