# backend/train.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
import shap
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = os.path.join("..", "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
SAVED_DIR = "saved_models"
os.makedirs(SAVED_DIR, exist_ok=True)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def engineer_features(df):
    """Create advanced features to improve model performance"""
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
    
    # NEW ADVANCED FEATURES
    # Engagement score
    df['EngagementScore'] = (df['JobInvolvement'] + df['JobSatisfaction'] + df['EnvironmentSatisfaction']) / 3
    
    # Compensation satisfaction
    df['CompensationSatisfaction'] = df['MonthlyIncome'] / (df['Age'] * 100)  # Normalized by age
    
    # Career stagnation indicator
    df['CareerStagnation'] = df['YearsSinceLastPromotion'] * df['YearsInCurrentRole']
    
    # Work-life pressure
    df['WorkLifePressure'] = (df['OverTime'] == 'Yes').astype(int) * (5 - df['WorkLifeBalance'])
    
    # Loyalty indicator
    df['LoyaltyScore'] = df['YearsAtCompany'] / (df['Age'] - 18 + 1)
    
    # Manager relationship
    df['ManagerRelationship'] = df['YearsWithCurrManager'] * df['RelationshipSatisfaction']
    
    # Income vs experience gap
    df['IncomeExperienceGap'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1) - df['MonthlyIncome'].median() / (df['TotalWorkingYears'].median() + 1)
    
    # Training effectiveness
    df['TrainingEffectiveness'] = df['TrainingTimesLastYear'] / (df['YearsAtCompany'] + 1)
    
    # Distance-overtime combo (high risk)
    df['CommuteStress'] = df['DistanceFromHome'] * (df['OverTime'] == 'Yes').astype(int) * (5 - df['WorkLifeBalance'])
    
    return df

def preprocess_and_train(df, compute_shap=True, shap_top_n=25, shap_sample_size=500):
    df = df.copy()
    
    # Feature engineering first
    print("Engineering features...")
    df = engineer_features(df)
    
    df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

    # drop ID column if exists
    if 'EmployeeNumber' in df.columns:
        df = df.drop(columns=['EmployeeNumber'])

    y = df['Attrition']
    X = df.drop(columns=['Attrition'])

    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object','category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Optimized models with better hyperparameters
    print("\n=== Training Optimized Models ===\n")
    
    # RandomForest with tuned params
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=8,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )
    
    # ExtraTrees (often better than RF)
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=8,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )
    
    # XGBoost with tuned params
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=4,
        gamma=0.1,
        min_child_weight=3,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    
    # LightGBM (often outperforms XGBoost)
    lgb = None
    if LGBMClassifier is not None:
        lgb = LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            num_leaves=31,
            min_child_samples=20,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
    
    # GradientBoosting
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=3,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    # Logistic Regression with balanced weights (best performer so far)
    lr = LogisticRegression(
        max_iter=3000,
        class_weight='balanced',
        C=0.08,
        solver='saga',
        penalty='elasticnet',
        l1_ratio=0.5,
        random_state=42
    )

    models = {
        'RandomForest': rf,
        'ExtraTrees': et,
        'XGBoost': xgb,
        **({'LightGBM': lgb} if lgb is not None else {}),
        'GradientBoosting': gb,
        'LogisticRegression': lr
    }

    best = {'name': None, 'auc': -1, 'model': None, 'pipeline': None}
    results = {}
    trained_models = {}

    for name, clf in models.items():
        # Try different sampling strategies for different models
        if name in ['LogisticRegression', 'GradientBoosting']:
            sampling_strategy = ADASYN(random_state=42, n_neighbors=3)
        else:
            sampling_strategy = SMOTETomek(random_state=42)
            
        pipe = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('sampling', sampling_strategy),
            ('clf', clf)
        ])

        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
        y_pred = pipe.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"  → AUC: {auc:.4f}")
        
        results[name] = {
            'roc_auc': float(auc),
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        trained_models[name] = pipe

        if auc > best['auc']:
            best['name'] = name
            best['auc'] = auc
            best['model'] = pipe
            best['pipeline'] = pipe
    
    # Create STACKING ensemble (more powerful than voting)
    print("\n=== Creating Stacking Ensemble ===")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)[:4]
    
    # Base estimators for stacking
    base_estimators = [(name, trained_models[name]) for name, _ in sorted_models]
    
    # Meta-learner (final estimator)
    meta_learner = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        C=0.1,
        random_state=42
    )
    
    # Stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    print("Training Stacking Ensemble (this may take a moment)...")
    stacking_clf.fit(X_train, y_train)
    probs = stacking_clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)
    y_pred = stacking_clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"  → Stacking AUC: {auc:.4f}")
    
    results['StackingEnsemble'] = {
        'roc_auc': float(auc),
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    if auc > best['auc']:
        best['name'] = 'StackingEnsemble'
        best['auc'] = auc
        best['model'] = stacking_clf
        best['pipeline'] = stacking_clf
    
    # Weighted voting based on performance
    print("\n=== Creating Weighted Voting Ensemble ===")
    
    # Calculate weights based on AUC scores
    weights = [results[name]['roc_auc'] for name, _ in base_estimators]
    
    voting_clf = VotingClassifier(
        estimators=base_estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )
    
    print("Training Weighted Voting Ensemble...")
    voting_clf.fit(X_train, y_train)
    probs = voting_clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)
    y_pred = voting_clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"  → Weighted Voting AUC: {auc:.4f}")
    
    results['WeightedVotingEnsemble'] = {
        'roc_auc': float(auc),
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    if auc > best['auc']:
        best['name'] = 'WeightedVotingEnsemble'
        best['auc'] = auc
        best['model'] = voting_clf
        best['pipeline'] = voting_clf

    pipeline_path = os.path.join(SAVED_DIR, 'pipeline.pkl')
    model_path = os.path.join(SAVED_DIR, 'best_model.pkl')
    # Use compression to reduce file size
    joblib.dump(best['pipeline'], pipeline_path, compress=3)
    joblib.dump(best['pipeline'], model_path, compress=3)
    print(f"\nModel saved with compression to: {pipeline_path}")

    # -----------------------------------------------------------
    # SAVE SCHEMA WITH DEFAULT VALUES FOR PREDICTION TIME
    # -----------------------------------------------------------
    schema = {
        "numeric_defaults": {},
        "categorical_defaults": {},
        "all_features": numeric_features + cat_features
    }

    for col in numeric_features:
        schema["numeric_defaults"][col] = float(X[col].median())

    for col in cat_features:
        schema["categorical_defaults"][col] = str(X[col].mode()[0])

    with open(os.path.join(SAVED_DIR, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    # -----------------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------------
    with open(os.path.join(SAVED_DIR, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # -----------------------------------------------------------
    # SAVE FEATURE STATS FOR LIME BACKGROUND SAMPLING
    # -----------------------------------------------------------
    try:
        feature_stats = {
            "numeric": {},
            "categorical": {}
        }
        for col in numeric_features:
            s = X[col].astype(float)
            feature_stats["numeric"][col] = {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0))
            }
        for col in cat_features:
            vc = X[col].astype(str).value_counts(normalize=True)
            feature_stats["categorical"][col] = {str(k): float(v) for k, v in vc.items()}
        with open(os.path.join(SAVED_DIR, 'feature_stats.json'), 'w') as f:
            json.dump(feature_stats, f, indent=2)
        print("Feature stats saved for LIME background sampling.")
    except Exception as e:
        print("Warning: failed to save feature stats:", str(e))

    # -----------------------------------------------------------
    # COMPUTE GLOBAL SHAP SUMMARY (mean absolute SHAP)
    # -----------------------------------------------------------
    if compute_shap:
        try:
            print("Computing global SHAP summary (this may take a bit)...")

            final_pipeline = best['pipeline']
            
            # Handle ensemble vs single model
            if 'Ensemble' in best['name']:
                # Use the best individual model for SHAP (first estimator in ensemble)
                shap_model_name = sorted_models[0][0]
                shap_pipeline = trained_models[shap_model_name]
                preprocessor = shap_pipeline.named_steps['preprocessor']
                estimator = shap_pipeline.named_steps['clf']
                print(f"  Using {shap_model_name} for SHAP computation...")
            else:
                preprocessor = final_pipeline.named_steps['preprocessor']
                estimator = final_pipeline.named_steps['clf']

            sample_X = X.sample(n=min(shap_sample_size, len(X)), random_state=42)
            X_trans = preprocessor.transform(sample_X)
            X_trans = np.asarray(X_trans, dtype=np.float64)

            est_name = estimator.__class__.__name__.lower()

            if ('xgb' in est_name) or ('randomforest' in est_name) or ('gradientboosting' in est_name):
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_trans)
            elif 'logisticregression' in est_name:
                explainer = shap.LinearExplainer(estimator, X_trans)
                shap_values = explainer.shap_values(X_trans)
            else:
                explainer = shap.KernelExplainer(lambda Z: estimator.predict_proba(Z)[:, 1], X_trans[:50])
                shap_values = explainer.shap_values(X_trans, nsamples=min(100, X_trans.shape[0]))

            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values

            mean_abs_shap = np.asarray(np.abs(sv).mean(axis=0)).ravel()

            try:
                num_feats = preprocessor.transformers_[0][2]
                cat_pipeline = preprocessor.transformers_[1][1]
                onehot = cat_pipeline.named_steps['onehot']
                original_cat_feats = preprocessor.transformers_[1][2]
                cat_feat_names = onehot.get_feature_names_out(original_cat_feats)
                all_feature_names = list(num_feats) + list(cat_feat_names)
            except:
                all_feature_names = [f"f{i}" for i in range(len(mean_abs_shap))]

            if len(all_feature_names) != len(mean_abs_shap):
                n = min(len(all_feature_names), len(mean_abs_shap))
                all_feature_names = all_feature_names[:n]
                mean_abs_shap = mean_abs_shap[:n]

            shap_df = pd.DataFrame({
                "feature": list(all_feature_names),
                "mean_abs_shap": list(map(float, mean_abs_shap))
            }).sort_values("mean_abs_shap", ascending=False)

            top_n = shap_df.head(shap_top_n).to_dict(orient="records")

            with open(os.path.join(SAVED_DIR, "shap_summary.json"), "w") as f:
                json.dump({"model": best['name'], "shap_top": top_n}, f, indent=2)

            print("SHAP summary saved.")
        
        except Exception as e:
            print("Error computing SHAP summary:", str(e))


    print(f"Best model: {best['name']} with AUC {best['auc']:.4f}")

    return best, results

if __name__ == "__main__":
    alt_path = os.path.join("data","WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if os.path.exists(alt_path):
        csv = alt_path
    elif os.path.exists(DATA_PATH):
        csv = DATA_PATH
    else:
        raise FileNotFoundError("CSV dataset not found. Place it in data/ directory.")

    df = load_data(csv)
    best, results = preprocess_and_train(df)
