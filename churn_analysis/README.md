# Churn Analysis Dashboard

A comprehensive churn prediction application built with Python, XGBoost, and Streamlit. This project analyzes customer churn data to predict which customers are likely to leave, helping businesses retain valuable customers.

## Features

- **XGBoost Model Training**: Automated pipeline for training churn prediction models on customer data
- **Multiple Dashboard Versions**:
  - `app.py`: Latest version with styled header and SVG icon
  - `dashboard.py`: Original dashboard with synthetic data generation
  - `main.py`: Batch prediction interface for CSV uploads
  - `main2.py`: Enhanced batch prediction with risk tiers
- **Docker Deployment**: Containerized application for easy deployment
- **Interactive Visualizations**: Streamlit-based web interface for predictions and analysis

## Technologies Used

- **Python 3.10**
- **XGBoost**: Machine learning model for classification
- **Streamlit**: Web framework for interactive dashboards
- **Pandas & NumPy**: Data manipulation and analysis
- **Joblib**: Model serialization
- **SHAP**: Feature importance explanations
- **Plotly**: Data visualizations
- **Docker**: Containerization

## Installation

### Local Setup

1. **Clone or download the repository**

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (optional, pre-trained model included):
   ```bash
   python classifier.py
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### Docker Setup

1. **Build the Docker image**:
   ```bash
   docker build -t churn-app .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 churn-app
   ```

3. **Access the app** at `http://localhost:8501`

## Usage

### Model Training
Run `classifier.py` to train the XGBoost model on the churn dataset. The script:
- Loads and preprocesses the data
- Trains the model with optimized hyperparameters
- Saves the model and feature list to `prediction_model.sav`

### Dashboard Features
- **Single Prediction**: Input customer features manually for churn probability
- **Batch Prediction**: Upload CSV files for bulk predictions
- **Risk Analysis**: View customers by risk tiers (Low, Medium, High)
- **Feature Importance**: Understand which factors influence churn predictions

## Data

The model is trained on customer churn data with features including:
- Account length
- International plan
- Voice mail plan
- Call statistics (minutes, calls, charges)
- Customer service calls

## Model Performance

- **Algorithm**: XGBoost Classifier
- **Evaluation Metric**: AUC (Area Under Curve)
- **Hyperparameters**: Optimized for imbalanced data with scale_pos_weight

## Project Structure

```
├── app.py                 # Main Streamlit dashboard
├── classifier.py          # Model training pipeline
├── dashboard.py           # Original dashboard
├── main.py               # Batch prediction app
├── main2.py              # Enhanced batch prediction
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── prediction_model.sav  # Trained model artifact
├── churn.csv            # Training data
├── pic.svg              # Icon for dashboard
└── README.md            # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

