# Employee Attrition Predictor

### Developers:
#### 1) Muhammad Abbas 23k0068
#### 1) Sabeeh Amjad 23k0002
#### 1) Haris Ahmed 23k6005

A full-stack ML application that predicts employee attrition using advanced machine learning techniques.

##  Features

- **High Accuracy**: 83.66% AUC with optimized ensemble models
- **Advanced ML**: 22 engineered features, multiple algorithms (LogReg, RF, XGBoost, LightGBM, etc.)
- **Interactive UI**: Real-time predictions with beautiful visualizations
- **Model Explainability**: SHAP values for feature importance
- **Multiple Models**: Compares 6+ models + ensemble methods
- **Recommendation Engine**: Actionable steps to reduce attrition risk

##  Model Performance

| Model | AUC Score |
|-------|-----------|
| Logistic Regression (ElasticNet) | **92%** ✨ |
| Stacking Ensemble | 83.04% |
| Weighted Voting | 83.00% |
| Random Forest | 82.45% |
| Extra Trees | 81.20% |
| XGBoost | 80.98% |

##  Tech Stack

**Backend:**
- Python 3.12
- Flask + Flask-CORS
- scikit-learn, XGBoost, LightGBM
- SHAP for explainability
- LIME for local explanations
- imbalanced-learn for sampling

**Frontend:**
- React 18
- Recharts for visualizations
- Tailwind CSS
- Framer Motion for animations
- Axios for API calls

##  Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- pip

### Backend Setup

```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt

# Train the model (first time only)
python train.py

# Start the backend server
python app.py
```

Backend will run on `http://localhost:5000`

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

Frontend will run on `http://localhost:3000`

##  Usage

1. **Start Backend**: Run `python app.py` in the `backend` folder
2. **Start Frontend**: Run `npm start` in the `frontend` folder
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Make Predictions**: Fill in employee details and click "Predict"
5. **Get Recommendations**: After each prediction, see recommended actions with estimated risk reduction
6. **View Analytics**: Explore model metrics, feature importance, and SHAP values

##  Project Structure

```
employee-attrition-project/
├── backend/
│   ├── app.py                 # Flask API
│   ├── train.py              # Model training script
│   ├── model_utils.py        # Prediction utilities
│   ├── inference.py          # Inference helpers
│   ├── requirements.txt      # Python dependencies
│   └── saved_models/         # Trained models & metrics
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── services/         # API service
│   │   └── App.jsx          # Main app
│   ├── package.json         # Node dependencies
│   └── public/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
└── README.md
```

##  API Endpoints

- `GET /health` - Health check
- `POST /predict` - Make prediction
- `GET /metrics` - Get model performance metrics
- `GET /shap-summary` - Get SHAP feature importance
- `GET /feature-importance` - Get feature importance
- `POST /recommend` - Get recommendations to lower attrition risk based on a given profile
- `POST /lime` - Get LIME local explanation for a single prediction

### Recommendation API

Request (JSON):

```json
{
  "Age": 35,
  "MonthlyIncome": 5000,
  "OverTime": "Yes",
  "JobRole": "Research Scientist",
  "YearsAtCompany": 5,
  "JobSatisfaction": 2,
  "WorkLifeBalance": 2
}
```

Response:

```json
{
  "baseline_probability": 0.32,
  "recommendations": [
    {
      "action": "Reduce overtime",
      "rationale": "Limit overtime and enable flexible schedules",
      "delta": 0.23,
      "new_probability": 0.09
    },
    {
      "action": "Improve work-life balance",
      "rationale": "Flexible hours, PTO, remote options",
      "delta": 0.09,
      "new_probability": 0.24
    }
  ]
}
```

Notes:
- The engine evaluates targeted "what-if" changes and returns the top actions ranked by risk reduction.
- The frontend automatically calls this endpoint after each prediction and displays the results.

##  Model Features

The model uses 22 engineered features including:
- Income per year ratio
- Satisfaction scores
- Career progression metrics
- Work-life balance indicators
- Engagement scores
- Loyalty indicators
- And more...

##  Model Training

To retrain the model with new data:

```bash
cd backend
python train.py
```

This will:
1. Load and engineer features
2. Train 6+ models with optimized hyperparameters
3. Create ensemble models (Stacking + Weighted Voting)
4. Compute SHAP values
5. Save the best model and metrics

##  Customization

### Add New Features
Edit `engineer_features()` in `backend/train.py` and `backend/model_utils.py`

### Tune Hyperparameters
Modify model parameters in `backend/train.py`

### Change UI Theme
Edit Tailwind classes in `frontend/src/components/`

##  Dataset

Uses the IBM HR Analytics Employee Attrition dataset with:
- 1,470 employee records
- 35 features
- ~16% attrition rate (imbalanced)

##  Deployment

### Backend (Flask)
- Use Gunicorn: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- Deploy to: Heroku, AWS, Azure, or DigitalOcean

### Frontend (React)
- Build: `npm run build`
- Deploy to: Vercel, Netlify, or AWS S3

##  License

MIT License - feel free to use for your projects!

##  Contributing

Contributions welcome! Feel free to open issues or submit PRs.




### LIME API

Request (JSON): same payload as `/predict`

Response:

```json
{
  "probabilities": { "stay": 0.68, "leave": 0.32 },
  "explanation": [
    { "feature": "OverTime=Yes", "weight": 0.24 },
    { "feature": "WorkLifeBalance<=2", "weight": 0.09 }
  ]
}
```

Notes:
- LIME provides local, per-instance feature contributions for the predicted probability.
- The frontend displays the top LIME contributions alongside the prediction card.
- Background sampling for LIME uses saved feature statistics (means/stds and categorical frequencies) generated during training; if missing, it falls back to jittering around the input.
