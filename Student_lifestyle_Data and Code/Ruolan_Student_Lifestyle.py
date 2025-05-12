import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


# Load dataset
df = pd.read_csv('student_lifestyle_dataset..csv')

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display basic info about the dataset
print("\nDataset Info:")
print(df.info())

# Display statistical summary
print("\nDataset Description:")
print(df.describe())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nDataset Columns:")
print(df.columns.tolist())

# Check if there are missing values in each column
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Check for duplicated rows
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# Check unique values for categorical features
print("\nValue Counts for 'Stress_Level':")
print(df['Stress_Level'].value_counts())

print("\nValue Counts for 'Gender':")
print(df['Gender'].value_counts())

numeric_cols = [
    'Study_Hours_Per_Day',
    'Extracurricular_Hours_Per_Day',
    'Sleep_Hours_Per_Day',
    'Social_Hours_Per_Day',
    'Physical_Activity_Hours_Per_Day',
    'Grades'
]

# Plotting distributions of numeric features
print("\nPlotting histograms of numeric features...")
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Boxplot to identify outliers
print("\nBoxplots to inspect outliers...")
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# Correlation heatmap
print("\nCorrelation Heatmap:")
plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
# draw heatmap
ax = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
plt.title('Correlation Heatmap of Numeric Features', fontsize=14, pad=20)
plt.subplots_adjust(bottom=0.25, left=0.2, right=0.85)
plt.tight_layout
plt.show()

# Create a new categorical target variable based on Grades
def categorize_gpa(gpa):
    if gpa < 7.0:
        return 'Low'
    elif gpa < 8.5:
        return 'Medium'
    else:
        return 'High'

df['GPA_Level'] = df['Grades'].apply(categorize_gpa)

# Display value counts of the new label
print("\nGPA Level Distribution:")
print(df['GPA_Level'].value_counts())

# Plot the distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='GPA_Level', hue='GPA_Level', data=df, palette='Set2', order=['Low', 'Medium', 'High'], legend=False)
plt.title('Distribution of GPA Levels')
plt.xlabel('GPA Level')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Label Encoding for categorical variables
label_encoders = {}
for col in ['Gender', 'Stress_Level', 'GPA_Level']:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nLabel Encoded Columns:")
print(df[['Gender', 'Gender_Encoded', 'Stress_Level', 'Stress_Level_Encoded', 'GPA_Level', 'GPA_Level_Encoded']].head())

# Feature Scaling
scaler = StandardScaler()

features_to_scale = [
    'Study_Hours_Per_Day',
    'Extracurricular_Hours_Per_Day',
    'Sleep_Hours_Per_Day',
    'Social_Hours_Per_Day',
    'Physical_Activity_Hours_Per_Day'
]

df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print("\nScaled Numeric Features (first 5 rows):")
print(df_scaled[features_to_scale].head())

# =========================================== #
# Logistic Regression Training and Evaluation #
# =========================================== #

# Select feature columns and target column
X = df_scaled[[
    'Study_Hours_Per_Day',                # Study Hours
    'Extracurricular_Hours_Per_Day',      # Extracurricular Hours
    'Sleep_Hours_Per_Day',                # Sleep Hours
    'Social_Hours_Per_Day',               # Social Hours
    'Physical_Activity_Hours_Per_Day',    # Physical Activity
    'Gender_Encoded',                     # Encoded Gender
    'Stress_Level_Encoded'                # Encoded Stress Level
]]

y = df_scaled['GPA_Level_Encoded']  # Target variable GPA Level

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Binarize the output for multi-class ROC computation
y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_train_bin.shape[1]

# Create and fit the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Print evaluation metrics
print("\n=== Logistic Regression Evaluation ===")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoders['GPA_Level'].classes_
))

# ROC & AUC for Logistic Regression
lr_ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr_ovr.fit(X_train, y_train_bin)
lr_score = lr_ovr.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], lr_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Logistic Regression")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# ======================== #
# Decision Tree Classifier #
# ======================== #

# Create and train the Decision Tree model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Predict on the test set
tree_pred = tree_model.predict(X_test)

# Print evaluation metrics
print("\n=== Decision Tree Evaluation ===")
print("Accuracy Score:", accuracy_score(y_test, tree_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, tree_pred,
    target_names=label_encoders['GPA_Level'].classes_
))

# ROC & AUC for Decision Tree
tree_ovr = OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
tree_ovr.fit(X_train, y_train_bin)
tree_score = tree_ovr.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], tree_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Decision Tree")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()


# ======================== #
# Random Forest Classifier #
# ======================== #

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
rf_pred = rf_model.predict(X_test)

# Print evaluation metrics
print("\n=== Random Forest Evaluation ===")
print("Accuracy Score:", accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, rf_pred,
    target_names=label_encoders['GPA_Level'].classes_
))

# ROC & AUC for Random Forest
rf_ovr = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf_ovr.fit(X_train, y_train_bin)
rf_score = rf_ovr.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Random Forest")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# =================== #
# K-Nearest Neighbors #
# =================== #

# Create and train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict on the test set
knn_pred = knn_model.predict(X_test)

# Print evaluation metrics
print("\n=== K-Nearest Neighbors Evaluation ===")
print("Accuracy Score:", accuracy_score(y_test, knn_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, knn_pred,
    target_names=label_encoders['GPA_Level'].classes_
))

# ROC & AUC for KNN
knn_ovr = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
knn_ovr.fit(X_train, y_train_bin)
knn_score = knn_ovr.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], knn_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - KNN")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# ============================== #
# Compare Model Accuracy Results #
# ============================== #

# Collect model names and their corresponding accuracy scores
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN']
accuracy_scores = [
    accuracy_score(y_test, y_pred),
    accuracy_score(y_test, tree_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, knn_pred)
]

# Create a DataFrame to display the scores
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracy_scores
})

# Print as table
print("\nModel Comparison (Accuracy):")
print(results_df)

# Plot the comparison as bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', hue='Model', data=results_df, palette='pastel', legend=False)
plt.title('Accuracy Comparison of Models')

for index, row in results_df.iterrows():
    plt.text(x=index,
             y=row['Accuracy'] + 0.01,
             s=f"{row['Accuracy']:.2f}",
             ha='center', va='bottom', fontsize=10)

plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# ============================== #
# Tuned Random Forest Evaluation #
# ============================== #

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize the model
tuned_rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=tuned_rf, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1)

# Fit on training data
grid_search.fit(X_train, y_train)

# Get best model
best_rf_model = grid_search.best_estimator_

# Predict on test data
tuned_rf_pred = best_rf_model.predict(X_test)

# Evaluation
print("\n=== Tuned Random Forest Evaluation ===")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy Score:", accuracy_score(y_test, tuned_rf_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, tuned_rf_pred,
    target_names=label_encoders['GPA_Level'].classes_
))

# ROC & AUC for Tuned Random Forest
tuned_rf_ovr = OneVsRestClassifier(best_rf_model)
tuned_rf_ovr.fit(X_train, y_train_bin)
tuned_rf_score = tuned_rf_ovr.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], tuned_rf_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Tuned Random Forest")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# ====================================== #
# Compare Default vs Tuned Random Forest #
# ====================================== #

rf_default_acc = accuracy_score(y_test, rf_pred)
rf_tuned_acc = accuracy_score(y_test, tuned_rf_pred)

# create dataframe with both model accuracies
rf_compare_df = pd.DataFrame({
    'Model': ['Default RF', 'Tuned RF'],
    'Accuracy': [rf_default_acc, rf_tuned_acc]
})

print("\nRandom Forest (Default vs Tuned) Accuracy Comparison:")
print(rf_compare_df)

# plot bar chart comparing accuracies
plt.figure(figsize=(6, 5))
sns.barplot(x='Model', y='Accuracy', data=rf_compare_df, palette='spring')

# add text labels above each bar
for index, row in rf_compare_df.iterrows():
    plt.text(x=index,
             y=row['Accuracy'] + 0.01,
             s=f"{row['Accuracy']:.2f}",
             ha='center', va='bottom', fontsize=10)

plt.ylim(0, 1)
plt.title('Default vs Tuned Random Forest Accuracy')
plt.tight_layout()
plt.show()