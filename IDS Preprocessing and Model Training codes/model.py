import pandas as pd
import time
import tracemalloc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('cybersecurity_intrusion_data.csv')

# Define feature columns and target
num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_features.remove('attack_detected')
cat_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove session_id if present
if 'session_id' in cat_features:
    cat_features.remove('session_id')

# Prepare features (all features + target)
X = df[num_features + cat_features]
y = df['attack_detected']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='mean'), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Models and param grids
models_params = {
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2, 5]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.1, 0.01]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__num_leaves': [31, 50],
            'classifier__learning_rate': [0.1, 0.01]
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
    }
}

def estimate_min_data_points(name, X, params):
    n_features = X.shape[1]
    n_estimators = params.get('classifier__n_estimators', 100)
    if isinstance(n_estimators, int):
        n_estimators_val = n_estimators
    elif isinstance(n_estimators, (list, tuple)):
        n_estimators_val = max(n_estimators)
    else:
        n_estimators_val = 100
    if name in ['Random Forest', 'XGBoost', 'LightGBM']:
        return 10 * n_features * n_estimators_val
    elif name == 'Logistic Regression':
        return 10 * n_features
    else:
        return 100

# Training models and storing best estimators
results_metrics = {}
best_models = {}

for name, info in models_params.items():
    print(f"\nTraining {name}")
    model = info['model']
    param_grid = info['params']
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', model)
    ])
    tracemalloc.start()
    start_time = time.time()
    grid = GridSearchCV(pipeline, param_grid=param_grid, scoring=make_scorer(f1_score), cv=5, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = time.time() - start_time
    mem_peak_mb = mem_peak / (1024**2)
    best_model = grid.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_score = best_model.predict_proba(X_test)[:, 1]
    roc_auc = auc(*roc_curve(y_test, y_score)[:2])

    results_metrics[name] = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'time': elapsed_time,
        'memory': mem_peak_mb,
        'min_data_points': estimate_min_data_points(name, X_train, grid.best_params_)
    }

# Print results
for name, metrics in results_metrics.items():
    print(f"\n{name} Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Training Time: {metrics['time']:.2f} seconds")
    print(f"Peak Memory: {metrics['memory']:.2f} MB")
    print(f"Estimated Min Data Points Needed: {metrics['min_data_points']:.0f}")

# Plot ROC curves for each best model
plt.figure(figsize=(10, 8))
for name, model in best_models.items():
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of Classifiers")
plt.legend()
plt.show()
