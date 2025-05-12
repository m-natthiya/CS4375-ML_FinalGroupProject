import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc


# Load dataset
df = pd.read_csv('ai_ghibli_trend_dataset_v2.csv')

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

# Select numeric columns to visualize
numeric_cols = [
    'likes',
    'shares',
    'comments',
    'generation_time',
    'gpu_usage',
    'file_size_kb',
    'style_accuracy_score'
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
ax = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
plt.title('Correlation Heatmap of Numeric Features', fontsize=14, pad=20)
plt.subplots_adjust(bottom=0.25, left=0.2, right=0.85)
plt.tight_layout()
plt.show()

# Convert style_accuracy_score into 3 level classification target
def categorize_style(score):
    if score < 65:
        return 'Low'
    elif score < 85:
        return 'Medium'
    else:
        return 'High'

df['Style_Level'] = df['style_accuracy_score'].apply(categorize_style)

# Display value counts of the new label
print("\nStyle Level Distribution:")
print(df['Style_Level'].value_counts())

# Plot distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Style_Level', hue='Style_Level', data=df, palette='Set2', order=['Low', 'Medium', 'High'], legend=False)
plt.title('Distribution of Style Levels')
plt.xlabel('Style Level')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Encode categorial columns for modeling
label_encoders = {}
for col in ['platform', 'is_hand_edited', 'ethical_concerns_flag', 'Style_Level']:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nEncoded Category Columns (Preview):")
print(df[['platform', 'platform_Encoded', 'is_hand_edited', 'is_hand_edited_Encoded', 'Style_Level', 'Style_Level_Encoded']].head())

# Split resolution column into width and height as numeric
df[['width', 'height']] = df['resolution'].str.split('x', expand=True).astype(int)

# Normalize numeric columns
features_to_scale = [
    'likes',
    'shares',
    'comments',
    'generation_time',
    'gpu_usage',
    'file_size_kb',
    'width',
    'height'
]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print("\nScaled Numeric Features (first 5 rows):")
print(df_scaled[features_to_scale].head())

# =========================================== #
# Logistic Regression Training and Evaluation #
# =========================================== #

# Select feature columns and target column
X = df_scaled[[
    'likes',
    'shares',
    'comments',
    'generation_time',
    'gpu_usage',
    'file_size_kb',
    'width',
    'height',
    'platform_Encoded',
    'is_hand_edited_Encoded',
    'ethical_concerns_flag_Encoded'
]]

y = df_scaled['Style_Level_Encoded']    # Target variable Style Level

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# For ROC-AUC (multi-class)
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
    target_names=label_encoders['Style_Level'].classes_
))

# ROC & AUC for Logistic Regression
lr_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr_clf.fit(X_train, y_train_bin)
lr_score = lr_clf.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], lr_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Logistic Regression (Multi-Class)")
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
    target_names=label_encoders['Style_Level'].classes_
))

# ROC & AUC for Multi-Class Decision Tree
tree_ovr = OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
tree_ovr.fit(X_train, y_train_bin)
tree_score = tree_ovr.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], tree_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Decision Tree (Multi-Class)")
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
    target_names=label_encoders['Style_Level'].classes_
))

# ROC & AUC for Multi-Class Random Forest
rf_ovr = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf_ovr.fit(X_train, y_train_bin)
rf_score = rf_ovr.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Random Forest (Multi-Class)")
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
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, knn_pred))
print("\nClassification Report:")
print(classification_report(
    y_test, knn_pred,
    target_names=label_encoders['Style_Level'].classes_
))

# ROC & AUC for Multi-Class KNN
knn_ovr = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
knn_ovr.fit(X_train, y_train_bin)
knn_score = knn_ovr.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], knn_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - KNN (Multi-Class)")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()


# ========================= #
# Support Vector Classifier #
# ========================= #

# Create and train the Support Vector Classifier
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Predict on test data
svm_pred = svm_model.predict(X_test)

# Print evaluation results
print("\n=== Support Vector Machine Evaluation ===")
print("Accuracy Score:", accuracy_score(y_test, svm_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, svm_pred,
    target_names=label_encoders['Style_Level'].classes_
))

# ROC & AUC for Multi-Class SVM
svm_ovr = OneVsRestClassifier(SVC(probability=True, random_state=42))
svm_ovr.fit(X_train, y_train_bin)
svm_score = svm_ovr.predict_proba(X_test)

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], svm_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - SVM (Multi-Class)")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()


# === 4 Model Training New Version === #

# ============================================ #
# Binary Classification: Good vs. Poor Styling #
# ============================================ #

# Create new binary classification label
# set style_accuracy_score >= 75 as 'High', otherwise 'Low'
df['Style_Binary'] = df['style_accuracy_score'].apply(lambda x: 'High' if x >= 75 else 'Low')

# Encode binary target label
le_binary = LabelEncoder()
df['Style_Binary_Encoded'] = le_binary.fit_transform(df['Style_Binary'])  # 'High'=1, 'Low'=0
df_scaled['Style_Binary_Encoded'] = df['Style_Binary_Encoded']

print("\n[Binary] Style Classification Distribution:")
print(df['Style_Binary'].value_counts())

# Split features and target
X_bin = df_scaled[[
    'likes',
    'shares',
    'comments',
    'generation_time',
    'gpu_usage',
    'file_size_kb',
    'width',
    'height',
    'platform_Encoded',
    'is_hand_edited_Encoded',
    'ethical_concerns_flag_Encoded'
]]

y_bin = df_scaled['Style_Binary_Encoded']

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

# ============================================= #
# Logistic Regression for Binary Classification #
# ============================================= #

print("\n=== [Binary] Logistic Regression ===")
log_bin = LogisticRegression(max_iter=1000)
log_bin.fit(Xb_train, yb_train)
log_pred = log_bin.predict(Xb_test)

print("Accuracy Score:", accuracy_score(yb_test, log_pred))

print("Classification Report:")
print(classification_report(yb_test, log_pred, target_names=le_binary.classes_))

# ROC & AUC for [Binary] Logistic Regression
log_bin_proba = log_bin.predict_proba(Xb_test)[:, 1]

fpr, tpr, _ = roc_curve(yb_test, log_bin_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - [Binary] Logistic Regression")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()


# ======================================= #
# Decision Tree for Binary Classification #
# ======================================= #

print("\n=== [Binary] Decision Tree ===")
tree_bin = DecisionTreeClassifier(random_state=42)
tree_bin.fit(Xb_train, yb_train)
tree_pred = tree_bin.predict(Xb_test)

print("Accuracy Score:", accuracy_score(yb_test, tree_pred))

print("Classification Report:")
print(classification_report(yb_test, tree_pred, target_names=le_binary.classes_))

# ROC & AUC for [Binary] Decision Tree
tree_bin_proba = tree_bin.predict_proba(Xb_test)[:, 1]

fpr, tpr, _ = roc_curve(yb_test, tree_bin_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - [Binary] Decision Tree")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()


# ======================================= #
# Random Forest for Binary Classification #
# ======================================= #

print("\n=== [Binary] Random Forest ===")
rf_bin = RandomForestClassifier(random_state=42)
rf_bin.fit(Xb_train, yb_train)
rf_pred = rf_bin.predict(Xb_test)

print("Accuracy Score:", accuracy_score(yb_test, rf_pred))

print("Classification Report:")
print(classification_report(yb_test, rf_pred, target_names=le_binary.classes_))

# ROC & AUC for [Binary] Random Forest
rf_bin_proba = rf_bin.predict_proba(Xb_test)[:, 1]

fpr, tpr, _ = roc_curve(yb_test, rf_bin_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - [Binary] Random Forest")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()


# ============================= #
# KNN for Binary Classification #
# ============================= #

print("\n=== [Binary] K-Nearest Neighbors ===")
knn_bin = KNeighborsClassifier(n_neighbors=5)
knn_bin.fit(Xb_train, yb_train)
knn_pred = knn_bin.predict(Xb_test)

print("Accuracy Score:", accuracy_score(yb_test, knn_pred))

print("Classification Report:")
print(classification_report(yb_test, knn_pred, target_names=le_binary.classes_))

# ROC & AUC for [Binary] KNN
knn_bin_proba = knn_bin.predict_proba(Xb_test)[:, 1]

fpr, tpr, _ = roc_curve(yb_test, knn_bin_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - [Binary] K-Nearest Neighbors")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()


# =================================================== #
# Support Vector Classifier for Binary Classification #
# =================================================== #

print("\n=== [Binary] Support Vector Machine ===")
svm_binary_model = SVC(probability=True, random_state=42)
svm_binary_model.fit(Xb_train, yb_train)
svm_binary_pred = svm_binary_model.predict(Xb_test)

print("Accuracy Score:", accuracy_score(yb_test, svm_binary_pred))

print("Classification Report:")
print(classification_report(yb_test, svm_binary_pred, target_names=le_binary.classes_))

# Binary Style Level Count Plot

plt.figure(figsize=(6, 4))

# plot the distribution of binary style levels
sns.countplot(x='Style_Binary', hue='Style_Binary', data=df, palette='Set2', legend=False)

plt.title('Distribution of [Binary] Style Levels')
plt.xlabel('Style Level (Binary)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# ROC & AUC for [Binary] Support Vector Machine
svm_bin_proba = svm_binary_model.predict_proba(Xb_test)[:, 1]

fpr, tpr, _ = roc_curve(yb_test, svm_bin_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - [Binary] SVM")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()


# ============================== #
# Compare Model Accuracy Results #
# ============================== #

# Collect accuracy scores for multi-class classification
multi_model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'SVM']
multi_accuracies = [
    accuracy_score(y_test, y_pred),        # Logistic
    accuracy_score(y_test, tree_pred),     # Tree
    accuracy_score(y_test, rf_pred),       # RF
    accuracy_score(y_test, knn_pred),      # KNN
    accuracy_score(y_test, svm_pred)       # SVM
]

# Collect accuracy scores for binary classification
binary_model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'SVM']
binary_accuracies = [
    accuracy_score(yb_test, log_pred),
    accuracy_score(yb_test, tree_pred),
    accuracy_score(yb_test, rf_pred),
    accuracy_score(yb_test, knn_pred),
    accuracy_score(yb_test, svm_binary_pred)
]

# Plot 1: Accuracy of models under multi-class classification
multi_df = pd.DataFrame({'Model': multi_model_names, 'Accuracy': multi_accuracies})
print("\n[Multi-Class] Model Accuracy Comparison:")
print(multi_df)

plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', hue='Model', data=multi_df, palette='Set2', legend=False)
plt.title('Accuracy Comparison - Multi-Class Classification')
plt.ylim(0, 1)
for i, acc in enumerate(multi_accuracies):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
plt.tight_layout()
plt.show()

# Plot 2: Accuracy of models under binary classification
binary_df = pd.DataFrame({'Model': binary_model_names, 'Accuracy': binary_accuracies})
print("\n[Binary] Model Accuracy Comparison:")
print(binary_df)

plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', hue='Model', data=binary_df, palette='Set1', legend=False)
plt.title('Accuracy Comparison - Binary Classification')
plt.ylim(0, 1)
for i, acc in enumerate(binary_accuracies):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
plt.tight_layout()
plt.show()

# Plot 3: Side-by-side comparison of each model's accuracy across both settings
compare_df = pd.DataFrame({
    'Model': multi_model_names,
    'Multi-Class': multi_accuracies,
    'Binary': binary_accuracies
})

compare_df_melted = pd.melt(compare_df, id_vars='Model', value_vars=['Multi-Class', 'Binary'],
                            var_name='Classification Type', value_name='Accuracy')

plt.figure(figsize=(10, 6))
sns.set(style='whitegrid')
ax = sns.barplot(x='Model', y='Accuracy', hue='Classification Type',
                 data=compare_df_melted, palette='coolwarm')

plt.title('Model Accuracy: Multi-Class vs Binary')
plt.ylim(0, 1)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=9)

plt.tight_layout()
plt.show()

# ======================================================== #
# Tuned Random Forest Evaluation for Binary Classification #
# ======================================================== #

# Define hyperparameter grid
param_grid_bin = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize the model
tuned_rf_bin = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search_bin = GridSearchCV(estimator=tuned_rf_bin,
                                param_grid=param_grid_bin,
                                cv=5, scoring='accuracy',
                                n_jobs=-1)

# Fit on training data
grid_search_bin.fit(Xb_train, yb_train)

# Get best model
best_rf_bin_model = grid_search_bin.best_estimator_

# Predict on test data
tuned_rf_bin_pred = best_rf_bin_model.predict(Xb_test)

# Evaluation
print("\n=== [Binary] Tuned Random Forest Evaluation ===")
print("Best Parameters:", grid_search_bin.best_params_)
print("Accuracy Score:", accuracy_score(yb_test, tuned_rf_bin_pred))

print("\nClassification Report:")
print(classification_report(
    yb_test, tuned_rf_bin_pred,
    target_names=le_binary.classes_
))

# ROC & AUC for [Binary] Tuned Random Forest
tuned_rf_bin_proba = best_rf_bin_model.predict_proba(Xb_test)[:, 1]

fpr, tpr, _ = roc_curve(yb_test, tuned_rf_bin_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - [Binary] Tuned Random Forest")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# ====================================== #
# Compare Default vs Tuned Random Forest #
# ====================================== #

rf_default_bin_acc = accuracy_score(yb_test, rf_pred)
rf_tuned_bin_acc = accuracy_score(yb_test, tuned_rf_bin_pred)

# Create dataframe with both model accuracies
rf_bin_compare_df = pd.DataFrame({
    'Model': ['Default RF (Binary)', 'Tuned RF (Binary)'],
    'Accuracy': [rf_default_bin_acc, rf_tuned_bin_acc]
})

print("\n[Binary] Random Forest (Default vs Tuned) Accuracy Comparison:")
print(rf_bin_compare_df)

# Plot bar chart comparing accuracies
plt.figure(figsize=(6, 5))
sns.barplot(x='Model', y='Accuracy', hue='Model', data=rf_bin_compare_df, palette='spring', legend=False)

# Add text labels above bars
for index, row in rf_bin_compare_df.iterrows():
    plt.text(x=index,
             y=row['Accuracy'] + 0.01,
             s=f"{row['Accuracy']:.2f}",
             ha='center', va='bottom', fontsize=10)

plt.ylim(0, 1)
plt.title('Default vs Tuned Random Forest Accuracy (Binary Classification)')
plt.tight_layout()
plt.show()




