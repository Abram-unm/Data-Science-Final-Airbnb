import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import time

#use LinearSVC for efficiency
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#setup paths
print("--- Script Starting ---")
SCRIPT_DIR = Path(__file__).parent 
PREPARED_DIR = SCRIPT_DIR.parent / 'prepared'
PLOTS_DIR = SCRIPT_DIR.parent / 'results_plots'
PLOTS_DIR.mkdir(exist_ok=True)

#load data
print(f"Looking for data in: {PREPARED_DIR}")

try:
    #use prepped files generated from preprocessed_rf.py
    train_df = pd.read_csv(PREPARED_DIR / 'train_preprocessed.csv')
    test_df = pd.read_csv(PREPARED_DIR / 'test_preprocessed.csv')
    print(f"SUCCESS: Loaded {train_df.shape[0]} training rows and {test_df.shape[0]} test rows.")
except FileNotFoundError:
    print("\nERROR: Could not find the CSV files!")
    print(f"Please check this folder: {PREPARED_DIR}")
    sys.exit()

#label Target (high occupancy)
target = "high_occupancy"

# Split Features/Target
X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_test = test_df.drop(columns=[target])
y_test = test_df[target]

#Train SVM 
print("\nTraining SVM Model (Linear Kernel)...")
print("Note: Pipeline will Impute missing values -> Scale -> Train SVM.")

start_time = time.time()  # <--- START CLOCK

#add imputer to handle missing values & replace with median, then do scaling
svm_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()), 
    ('svm', LinearSVC(random_state=42, dual=False, max_iter=5000))
])

svm_pipeline.fit(X_train, y_train)

end_time = time.time()    # <--- STOP CLOCK

print(f"Training Time: {end_time - start_time:.2f} seconds")
print("Training Complete!")

#evaluate
print("\nPredicting on Test Data (2022)...")
y_pred = svm_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {acc:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('Confusion Matrix: SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'confusion_matrix_svm.png')
plt.close()

#feature importance (coefficients)
print("Generating Feature Importance Plot...")

#extract coefficients from model
model = svm_pipeline.named_steps['svm']
coefs = model.coef_[0]
feature_names = X_train.columns

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
#sort by magnitude
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coef', ascending=False)

#plot top features
plt.figure(figsize=(10, 8))
top_features = coef_df.head(15)
sns.barplot(x='Coefficient', y='Feature', data=top_features, palette='coolwarm')
plt.title('How Features Affect Occupancy (SVM Coefficients)\nRight = Increases Occupancy, Left = Decreases')
plt.axvline(0, color='black', linestyle='--') 
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'feature_importance_svm.png')
plt.close()

print(f"\nSUCCESS! Plots saved to: {PLOTS_DIR}")