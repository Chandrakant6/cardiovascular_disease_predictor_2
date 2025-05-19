# cardiovascular_disease_predictor_2
improvement of cardiovascular_disease_predictor

# Heart Disease Prediction System

A machine learning-based system for predicting heart disease using various classification algorithms. This project implements multiple models and provides comprehensive analysis and visualization capabilities.

## Features

- **Multiple ML Models**:
  - Random Forest
  - Linear SVM
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Logistic Regression

- **Data Preprocessing**:
  - Age conversion (days to years)
  - BMI calculation
  - Outlier detection and handling
  - Feature scaling
  - Data sampling for efficient training

- **Analysis & Visualization**:
  - Correlation matrix
  - Key feature distributions
  - Confusion matrices for each model
  - Model comparison plots
  - Training time analysis

- **Model Evaluation**:
  - Cross-validation scores
  - Accuracy metrics
  - Training time tracking
  - Best model selection and saving

## Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd heart-disease-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset file (`cardio_train.csv`) in the project directory.

2. Run the script:
```bash
python heart_disease_predictor.py
```

3. The script will:
   - Load and preprocess the data
   - Train multiple models
   - Generate visualizations
   - Save the best performing model
   - Display performance metrics

## Output

The script generates several outputs:

1. **Visualizations** (in `visualizations/` directory):
   - `correlation_matrix.png`: Feature correlation analysis
   - `key_distributions.png`: Distribution of key features
   - `confusion_matrix_*.png`: Confusion matrices for each model
   - `model_comparison.png`: Comparison of model performances

2. **Saved Models** (in `saved_models/` directory):
   - Best performing model saved as `.joblib` file
   - Scaler saved for future predictions

3. **Console Output**:
   - Training progress
   - Model performance metrics
   - Cross-validation scores
   - Training times

## Model Performance

The models are evaluated based on:
- Accuracy
- Cross-validation scores
- Training time
- Confusion matrix

## Making Predictions

To make predictions with the saved model:

```python
from heart_disease_predictor import HeartDiseasePredictor

# Initialize predictor
predictor = HeartDiseasePredictor()

# Load the saved model
predictor.load_model("model_name")  # e.g., "random_forest"

# Prepare new data (should match the format of training data)
new_data = pd.DataFrame({
    'age': [...],
    'height': [...],
    'weight': [...],
    'ap_hi': [...],
    'ap_lo': [...],
    # ... other features
})

# Make predictions
predictions, probabilities = predictor.predict(new_data)
```

## Performance Optimization

The current implementation is optimized for speed by:
- Using 10% of the data for training
- Simplified model parameters
- Sequential training instead of parallel processing
- Reduced cross-validation folds

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
