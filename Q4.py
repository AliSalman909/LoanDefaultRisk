import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_csv('application_train.csv')  # Use the correct path here

df = df.dropna(axis=1, thresh=0.6 * len(df))  # Drop columns with >40% missing values
df = df.dropna()  # Drop rows with any missing values

cat_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop('TARGET', axis=1)
y = df['TARGET']  # 1 = default, 0 = no default

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# CatBoost
cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(X_train, y_train)

# Costs: FP = giving loan to risky person, FN = rejecting safe customer
false_positive_cost = 10000
false_negative_cost = 2000


def calculate_cost(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * false_positive_cost) + (fn * false_negative_cost)
    return total_cost

# Predict probabilities with best model (CatBoost)
y_probs = cat_model.predict_proba(X_test)[:, 1]

# Try different thresholds to minimize cost
thresholds = np.arange(0.0, 1.0, 0.01)
costs = [calculate_cost(y_test, y_probs, t) for t in thresholds]

best_threshold = thresholds[np.argmin(costs)]
print(f'Optimal threshold: {best_threshold:.2f}, Minimum Cost: ${min(costs):,.2f}')

# Final predictions
final_preds = (y_probs >= best_threshold).astype(int)

print(classification_report(y_test, final_preds))


feat_importance = pd.Series(cat_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:20]
plt.figure(figsize=(10,6))
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title("Top 20 Feature Importances")
plt.show()
