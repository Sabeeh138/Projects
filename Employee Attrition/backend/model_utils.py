# backend/model_utils.py
import joblib
import os
import json
import pandas as pd
import numpy as np

SAVED_DIR = "saved_models"
PIPELINE_PATH = os.path.join(SAVED_DIR, "pipeline.pkl")

def engineer_features_single(df):
    """Apply same feature engineering as training"""
    df = df.copy()
    
    # Interaction features
    df['Income_Per_Year'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    df['Satisfaction_Balance_Score'] = df['JobSatisfaction'] * df['WorkLifeBalance']
    df['Total_Satisfaction'] = df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']
    
    # Age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['Young', 'Mid', 'Senior', 'Veteran'])
    
    # Income brackets
    df['IncomeLevel'] = pd.cut(df['MonthlyIncome'], bins=[0, 3000, 6000, 10000, 100000], 
                                labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # Experience ratios
    df['YearsAtCompany_Ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['YearsWithCurrManager_Ratio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    df['YearsSinceLastPromotion_Ratio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    
    # Work intensity
    df['WorkIntensity'] = (df['OverTime'] == 'Yes').astype(int) * df['JobInvolvement']
    
    # Career progression
    df['PromotionRate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['TrainingOpportunity'] = df['TrainingTimesLastYear'] * df['YearsAtCompany']
    
    # Distance impact
    df['DistanceImpact'] = df['DistanceFromHome'] * (df['OverTime'] == 'Yes').astype(int)
    
    # Stability score
    df['JobStability'] = df['YearsAtCompany'] * df['JobSatisfaction'] / (df['YearsSinceLastPromotion'] + 1)
    
    # NEW ADVANCED FEATURES (must match training)
    df['EngagementScore'] = (df['JobInvolvement'] + df['JobSatisfaction'] + df['EnvironmentSatisfaction']) / 3
    df['CompensationSatisfaction'] = df['MonthlyIncome'] / (df['Age'] * 100)
    df['CareerStagnation'] = df['YearsSinceLastPromotion'] * df['YearsInCurrentRole']
    df['WorkLifePressure'] = (df['OverTime'] == 'Yes').astype(int) * (5 - df['WorkLifeBalance'])
    df['LoyaltyScore'] = df['YearsAtCompany'] / (df['Age'] - 18 + 1)
    df['ManagerRelationship'] = df['YearsWithCurrManager'] * df['RelationshipSatisfaction']
    
    # For these features, we need median values from training - use safe defaults
    median_income = 4919.0  # approximate from dataset
    median_years = 7.0
    df['IncomeExperienceGap'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1) - median_income / (median_years + 1)
    df['TrainingEffectiveness'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 1)
    df['CommuteStress'] = df['DistanceFromHome'] * (df['OverTime'] == 'Yes').astype(int) * (5 - df['WorkLifeBalance'])
    
    return df

def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError("Pipeline not found. Run train.py first.")
    return joblib.load(PIPELINE_PATH)

def predict_single(user_input):
    pipeline = load_pipeline()

    # Load schema defaults
    schema_path = os.path.join(SAVED_DIR, "schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError("schema.json missing. Retrain the model.")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    all_features = schema["all_features"]
    numeric_defaults = schema["numeric_defaults"]
    categorical_defaults = schema["categorical_defaults"]

    # Convert user input into DataFrame
    X = pd.DataFrame([user_input])

    # Add missing columns with defaults
    for col in all_features:
        if col not in X.columns:
            if col in numeric_defaults:
                X[col] = numeric_defaults[col]
            elif col in categorical_defaults:
                X[col] = categorical_defaults[col]
            else:
                X[col] = None  # fallback

    # Ensure correct order
    X = X[all_features]
    
    # Apply feature engineering
    X = engineer_features_single(X)

    # Predict
    pred_proba = pipeline.predict_proba(X)[0, 1]
    pred = int(pred_proba >= 0.5)

    return {
        "probability": float(pred_proba),
        "prediction": pred
    }

def recommend_actions(user_input, max_actions=5):
    pipeline = load_pipeline()
    schema_path = os.path.join(SAVED_DIR, "schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError("schema.json missing. Retrain the model.")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    all_features = schema["all_features"]
    numeric_defaults = schema["numeric_defaults"]
    categorical_defaults = schema["categorical_defaults"]
    X = pd.DataFrame([user_input])
    for col in all_features:
        if col not in X.columns:
            if col in numeric_defaults:
                X[col] = numeric_defaults[col]
            elif col in categorical_defaults:
                X[col] = categorical_defaults[col]
            else:
                X[col] = None
    X = X[all_features]
    base_df = engineer_features_single(X.copy())
    base_p = float(pipeline.predict_proba(base_df)[0, 1])
    actions = []
    def test_change(changes, label, rationale):
        X2 = X.copy()
        for k, v in changes.items():
            X2[k] = v
        X2 = X2[all_features]
        X2 = engineer_features_single(X2)
        p2 = float(pipeline.predict_proba(X2)[0, 1])
        delta = base_p - p2
        if delta > 1e-3:
            actions.append({
                "action": label,
                "rationale": rationale,
                "delta": float(delta),
                "new_probability": float(p2),
                "changes": {k: user_input.get(k, None) for k in changes.keys()} | {f"to_{k}": changes[k] for k in changes.keys()}
            })
    v = X.iloc[0]
    if "OverTime" in X.columns and str(v.get("OverTime", "No")) == "Yes":
        test_change({"OverTime": "No"}, "Reduce overtime", "Limit overtime and enable flexible schedules")
    if "WorkLifeBalance" in X.columns:
        try:
            wlb = int(v.get("WorkLifeBalance", 3))
            if wlb < 4:
                test_change({"WorkLifeBalance": min(wlb + 1, 4)}, "Improve work-life balance", "Flexible hours, PTO, remote options")
        except:
            pass
    if "JobSatisfaction" in X.columns:
        try:
            js = int(v.get("JobSatisfaction", 3))
            if js < 4:
                test_change({"JobSatisfaction": min(js + 1, 4)}, "Boost job satisfaction", "Recognition, role clarity, meaningful work")
        except:
            pass
    if "EnvironmentSatisfaction" in X.columns:
        try:
            es = int(v.get("EnvironmentSatisfaction", 3))
            if es < 4:
                test_change({"EnvironmentSatisfaction": min(es + 1, 4)}, "Improve work environment", "Better tools, workspace and support")
        except:
            pass
    if "RelationshipSatisfaction" in X.columns:
        try:
            rs = int(v.get("RelationshipSatisfaction", 3))
            if rs < 4:
                test_change({"RelationshipSatisfaction": min(rs + 1, 4)}, "Strengthen manager relationship", "Feedback, mentorship, 1:1s")
        except:
            pass
    if "TrainingTimesLastYear" in X.columns:
        try:
            ttl = int(v.get("TrainingTimesLastYear", 0))
            if ttl < 4:
                test_change({"TrainingTimesLastYear": min(ttl + 2, 10)}, "Increase training", "Learning and development programs")
        except:
            pass
    if "YearsSinceLastPromotion" in X.columns:
        try:
            ysp = int(v.get("YearsSinceLastPromotion", 0))
            if ysp > 1:
                test_change({"YearsSinceLastPromotion": max(ysp - 1, 0)}, "Plan promotion/role growth", "Career path and advancement")
        except:
            pass
    if "DistanceFromHome" in X.columns:
        try:
            dfh = float(v.get("DistanceFromHome", 10.0))
            if dfh > 20:
                test_change({"DistanceFromHome": 10.0}, "Offer remote/hybrid", "Reduce commute burden")
        except:
            pass
    if "MonthlyIncome" in X.columns and "MonthlyIncome" in numeric_defaults:
        try:
            mi = float(v.get("MonthlyIncome", numeric_defaults["MonthlyIncome"]))
            med = float(numeric_defaults["MonthlyIncome"])
            if mi < med:
                test_change({"MonthlyIncome": med * 1.1}, "Adjust compensation", "Salary review or incentives")
        except:
            pass
    if "JobInvolvement" in X.columns:
        try:
            ji = int(v.get("JobInvolvement", 3))
            if ji < 4:
                test_change({"JobInvolvement": min(ji + 1, 4)}, "Enhance engagement", "Ownership, challenging projects")
        except:
            pass
    if "YearsWithCurrManager" in X.columns:
        try:
            ywcm = int(v.get("YearsWithCurrManager", 0))
            if ywcm < 2:
                test_change({"YearsWithCurrManager": min(ywcm + 1, 40)}, "Manager continuity", "Stable leadership and support")
        except:
            pass
    actions = sorted(actions, key=lambda x: x["delta"], reverse=True)[:max_actions]
    return {"baseline_probability": base_p, "recommendations": actions}

def lime_explain_single(user_input, num_features=10, num_samples=1000, bg_samples=500):
    from lime.lime_tabular import LimeTabularExplainer
    pipeline = load_pipeline()
    schema_path = os.path.join(SAVED_DIR, "schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError("schema.json missing. Retrain the model.")
    with open(schema_path, "r") as f:
        schema = json.load(f)

    pre = pipeline.named_steps.get('preprocessor')
    estimator = pipeline.named_steps.get('clf')
    if pre is None or estimator is None:
        raise RuntimeError("Pipeline missing preprocessor or classifier")

    # Build base input from schema and user input
    all_features = schema["all_features"]
    numeric_defaults = schema["numeric_defaults"]
    categorical_defaults = schema["categorical_defaults"]
    X = pd.DataFrame([user_input])
    for col in all_features:
        if col not in X.columns:
            if col in numeric_defaults:
                X[col] = numeric_defaults[col]
            elif col in categorical_defaults:
                X[col] = categorical_defaults[col]
            else:
                X[col] = None
    X = X[all_features]
    X_eng = engineer_features_single(X.copy())

    # Derive transformed feature names from preprocessor
    try:
        num_feats = list(pre.transformers_[0][2])
        cat_feats = list(pre.transformers_[1][2])
        onehot = pre.transformers_[1][1].named_steps['onehot']
        cat_names = list(onehot.get_feature_names_out(cat_feats))
        transformed_feature_names = num_feats + cat_names
    except Exception:
        transformed_feature_names = [f"f{i}" for i in range(pre.transform(X_eng.iloc[[0]]).shape[1])]

    # Try to use saved feature stats; else fallback to jitter around instance
    rng = np.random.default_rng(42)
    bg_rows = []
    base = X_eng.iloc[0].copy()
    numeric_cols = X_eng.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X_eng.select_dtypes(include=['object','category']).columns.tolist()

    stats_path = os.path.join(SAVED_DIR, 'feature_stats.json')
    stats = None
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        except Exception:
            stats = None

    for _ in range(bg_samples):
        row = base.copy()
        for c in numeric_cols:
            if stats and c in stats.get('numeric', {}):
                mean = stats['numeric'][c].get('mean', 0.0)
                std = stats['numeric'][c].get('std', 1.0)
                std = std if std > 1e-8 else max(0.1 * (abs(mean) + 1.0), 1e-3)
                row[c] = rng.normal(mean, std)
            else:
                val = float(row[c]) if pd.notnull(row[c]) else float(numeric_defaults.get(c, 0.0))
                sigma = 0.1 * (abs(val) + 1.0)
                row[c] = val + rng.normal(0, sigma)
        for c in categorical_cols:
            if stats and c in stats.get('categorical', {}):
                values = list(stats['categorical'][c].keys())
                probs = np.array(list(stats['categorical'][c].values()), dtype=float)
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(values)) / len(values)
                row[c] = rng.choice(values, p=probs)
            else:
                v = row[c]
                row[c] = v
        bg_rows.append(row)
    bg_df = pd.DataFrame(bg_rows)

    # Transform background and instance via preprocessor
    X_bg = pre.transform(bg_df)
    x_trans = pre.transform(X_eng.iloc[[0]])

    def predict_fn_trans(arr):
        return estimator.predict_proba(arr)

    explainer = LimeTabularExplainer(
        np.asarray(X_bg),
        feature_names=transformed_feature_names,
        class_names=["Stay", "Leave"],
        discretize_continuous=True,
        mode="classification"
    )

    exp = explainer.explain_instance(np.asarray(x_trans[0]), predict_fn_trans, num_features=num_features, num_samples=num_samples)
    probs = predict_fn_trans(x_trans)[0]
    pairs = exp.as_list()
    return {
        "probabilities": {"stay": float(probs[0]), "leave": float(probs[1])},
        "explanation": [{"feature": p[0], "weight": float(p[1])} for p in pairs]
    }
