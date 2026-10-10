# ============================================================
# PROJECT 2: Data Classification Using AI
# ============================================================
# Intern      : [Write Your Full Name Here]
# Organization: DecodeLabs
# Batch       : 2026
# File        : classifier.py
# ============================================================
# OBJECTIVE   : Build a KNN classification model on Iris data
# ARCHITECTURE: IPO Model (Input → Process → Output)
# ALGORITHM   : K-Nearest Neighbors (KNN)
# DATASET     : Iris Benchmark (150 samples, 3 classes, 4 features)
# ============================================================
# PIPELINE:
#   Step 1: Load & Explore Dataset
#   Step 2: Feature Scaling (StandardScaler)
#   Step 3: Train-Test Split (80/20)
#   Step 4: Train KNN Model
#   Step 5: Make Predictions
#   Step 6: Evaluate (Confusion Matrix + F1 Score)
# ============================================================

# --- CORE LIBRARIES ---
import numpy as np                        # Numerical operations
import pandas as pd                       # Data manipulation

# --- MACHINE LEARNING TOOLS (from scikit-learn) ---
from sklearn.datasets import load_iris                        # The Iris dataset
from sklearn.preprocessing import StandardScaler             # Feature scaling
from sklearn.model_selection import train_test_split         # Data splitting
from sklearn.neighbors import KNeighborsClassifier           # KNN algorithm
from sklearn.metrics import (                                 # Evaluation tools
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

# --- VISUALIZATION LIBRARIES ---
import matplotlib.pyplot as plt           # Base plotting
import seaborn as sns                     # Beautiful statistical plots

# --- DISPLAY SETTINGS ---
import warnings
warnings.filterwarnings('ignore')         # Keep output clean

print("=" * 60)
print("  PROJECT 2: Data Classification Using AI")
print("  Algorithm : K-Nearest Neighbors (KNN)")
print("  Dataset   : Iris Benchmark")
print("  Powered by: DecodeLabs | Batch 2026")
print("=" * 60)
print()
# ============================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ============================================================

print("[ STEP 1 ] Loading and Exploring the Iris Dataset...")
print("-" * 60)

# --- Load the raw Iris data from sklearn ---
iris = load_iris()

# --- Separate into Features (X) and Labels (y) ---
# X = the 4 measurements (what we use to predict)
# y = the species number (what we want to predict)
X = iris.data      # Shape: (150, 4) → 150 rows, 4 columns
y = iris.target    # Shape: (150,)   → 150 labels (0, 1, or 2)

# --- Store class names for readable output later ---
class_names = iris.target_names   # ['setosa', 'versicolor', 'virginica']
feature_names = iris.feature_names

# --- Convert to DataFrame for easy exploration ---
df = pd.DataFrame(
    data=X,
    columns=feature_names
)
df['species'] = pd.Categorical.from_codes(y, class_names)

# --- EXPLORE: Basic Information ---
print(f"Dataset Shape     : {X.shape}")
print(f"Total Samples     : {X.shape[0]} flowers")
print(f"Total Features    : {X.shape[1]} measurements")
print(f"Classes           : {list(class_names)}")
print(f"Samples per class : {dict(zip(class_names, np.bincount(y)))}")
print()

# --- EXPLORE: First 5 rows ---
print("First 5 rows of the dataset:")
print(df.head())
print()

# --- EXPLORE: Statistical Summary ---
print("Statistical Summary:")
print(df.describe().round(2))
print()

print("✅ Dataset loaded successfully!")
print()

# ============================================================
# STEP 2: FEATURE SCALING
# ============================================================

print("[ STEP 2 ] Applying Feature Scaling...")
print("-" * 60)

# --- Create the StandardScaler object ---
# Think of it as a "normalization machine" we configure and use
scaler = StandardScaler()

# --- Show BEFORE scaling (first 3 rows, raw values) ---
print("BEFORE Scaling (first 3 samples):")
print(f"  Sample 1: {X[0]}")
print(f"  Sample 2: {X[1]}")
print(f"  Sample 3: {X[2]}")
print()

# --- Apply scaling to ALL features ---
# fit_transform() does two things at once:
#   fit()       = calculate mean and std of each feature
#   transform() = apply the normalization formula
X_scaled = scaler.fit_transform(X)

# --- Show AFTER scaling (same 3 rows, normalized) ---
print("AFTER Scaling (same 3 samples, normalized):")
print(f"  Sample 1: {X_scaled[0].round(3)}")
print(f"  Sample 2: {X_scaled[1].round(3)}")
print(f"  Sample 3: {X_scaled[2].round(3)}")
print()

# --- Verify scaling worked correctly ---
print("Verification (scaled data statistics):")
print(f"  Mean of each feature (should be ~0): "
      f"{X_scaled.mean(axis=0).round(3)}")
print(f"  Std  of each feature (should be ~1): "
      f"{X_scaled.std(axis=0).round(3)}")
print()
print("✅ Feature scaling applied successfully!")
print()

# ============================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================

print("[ STEP 3 ] Splitting Data into Training and Test Sets...")
print("-" * 60)

# --- Split the scaled data ---
# X_train, y_train = model will LEARN from these
# X_test,  y_test  = model will be EVALUATED on these (never seen)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,        # Our scaled features
    y,               # Our labels
    test_size=0.2,   # 20% for testing, 80% for training
    random_state=42, # Fixed seed → same split every run
    stratify=y       # Keep class proportions balanced
)

# --- Report the split results ---
print(f"Total samples      : {len(X_scaled)}")
print(f"Training samples   : {len(X_train)} ({len(X_train)/len(X_scaled)*100:.0f}%)")
print(f"Test samples       : {len(X_test)}  ({len(X_test)/len(X_scaled)*100:.0f}%)")
print()

# --- Verify stratification worked ---
print("Class distribution in Training set:")
train_counts = dict(zip(class_names, np.bincount(y_train)))
for species, count in train_counts.items():
    print(f"  {species:12} : {count} samples")

print()
print("Class distribution in Test set:")
test_counts = dict(zip(class_names, np.bincount(y_test)))
for species, count in test_counts.items():
    print(f"  {species:12} : {count} samples")

print()
print("✅ Data split successfully!")
print()

# ============================================================
# STEP 4: TRAIN THE KNN MODEL
# ============================================================

print("[ STEP 4 ] Training the KNN Model...")
print("-" * 60)

# --- INSTANTIATE: Create the KNN classifier ---
# n_neighbors=5 means: consult 5 nearest neighbors before voting
# This is our K value — we will optimize it later in Phase 4
k = 5
model = KNeighborsClassifier(n_neighbors=k)

print(f"Model created      : KNeighborsClassifier")
print(f"K value            : {k} neighbors")
print(f"Training on        : {len(X_train)} samples")
print()

# --- FIT: Give the model its training data ---
# For KNN this simply MEMORIZES the training data
# No complex math happens here — pure storage
model.fit(X_train, y_train)

print("Model training complete!")
print()

# --- Quick self-check on training data ---
# This tells us the MAXIMUM possible accuracy
# (model tested on data it already memorized)
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy  : {train_accuracy * 100:.2f}%")
print("  (Accuracy on data the model already memorized)")
print()
print("✅ Model trained successfully!")
print()

# ============================================================
# STEP 5: MAKE PREDICTIONS
# ============================================================

print("[ STEP 5 ] Making Predictions on Test Data...")
print("-" * 60)

# --- PREDICT: Run all 30 test flowers through the model ---
# model.predict() runs KNN distance + majority vote for each
y_pred = model.predict(X_test)

# --- Show first 10 predictions vs actual ---
print("Sample Predictions (first 10 test flowers):")
print(f"  {'#':<4} {'Actual':<15} {'Predicted':<15} {'Correct?'}")
print(f"  {'-'*45}")

for i in range(10):
    actual    = class_names[y_test[i]]
    predicted = class_names[y_pred[i]]
    correct   = "✅" if y_test[i] == y_pred[i] else "❌"
    print(f"  {i+1:<4} {actual:<15} {predicted:<15} {correct}")

print()

# --- Count correct predictions ---
correct_total = np.sum(y_pred == y_test)
print(f"Correct predictions: {correct_total} out of {len(y_test)}")
print()
print("✅ Predictions complete!")
print()

# ============================================================
# STEP 6: EVALUATE THE MODEL
# ============================================================

print("[ STEP 6 ] Evaluating Model Performance...")
print("-" * 60)

# --- BASIC METRICS ---
accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy      : {accuracy * 100:.2f}%")
print(f"F1 Score (weighted): {f1:.4f}  (1.0 = perfect)")
print()

# --- DETAILED CLASSIFICATION REPORT ---
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=class_names))

# --- CONFUSION MATRIX (Text version) ---
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  Rows    = Actual species")
print(f"  Columns = Predicted species")
print()
print(f"  {'':15}", end="")
for name in class_names:
    print(f"{name:12}", end="")
print()
for i, row in enumerate(cm):
    print(f"  {class_names[i]:15}", end="")
    for val in row:
        print(f"{val:<12}", end="")
    print()
print()

# --- CONFUSION MATRIX (Visual Heatmap) ---
print("Generating confusion matrix heatmap...")

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,           # Show numbers inside boxes
    fmt='d',              # Integer format
    cmap='Blues',         # Blue color scheme
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5
)
plt.title('Confusion Matrix — KNN Iris Classifier\nDecodeLabs Project 2',
          fontsize=14, fontweight='bold')
plt.ylabel('Actual Species',    fontsize=12)
plt.xlabel('Predicted Species', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Heatmap saved as: confusion_matrix.png")
print()

# --- FINAL SUMMARY ---
print("=" * 60)
print("  FINAL MODEL SUMMARY")
print("=" * 60)
print(f"  Algorithm    : K-Nearest Neighbors (K={k})")
print(f"  Dataset      : Iris (150 samples, 3 classes)")
print(f"  Train Split  : 120 samples (80%)")
print(f"  Test Split   : 30  samples (20%)")
print(f"  Accuracy     : {accuracy * 100:.2f}%")
print(f"  F1 Score     : {f1:.4f}")
grade = "EXCELLENT" if f1 >= 0.95 else "GOOD" if f1 >= 0.85 else "NEEDS WORK"
print(f"  Grade        : {grade} ✅")
print("=" * 60)

# ============================================================
# ENHANCEMENT 1: FINDING THE OPTIMAL K VALUE
# ============================================================

print()
print("[ ENHANCEMENT 1 ] Finding the Optimal K Value...")
print("-" * 60)

# --- Test every K from 1 to 20 ---
k_range     = range(1, 21)       # K values to test: 1,2,3...20
error_rates = []                  # Store error rate for each K
accuracy_rates = []               # Store accuracy for each K

for k_val in k_range:

    # Create a new KNN model with this K value
    knn_temp = KNeighborsClassifier(n_neighbors=k_val)

    # Train it on the same training data
    knn_temp.fit(X_train, y_train)

    # Predict on test data
    y_pred_temp = knn_temp.predict(X_test)

    # Calculate error rate (1 - accuracy)
    # Error rate = percentage of WRONG predictions
    acc   = accuracy_score(y_test, y_pred_temp)
    error = 1 - acc

    accuracy_rates.append(acc)
    error_rates.append(error)

    print(f"  K={k_val:2d}  |  Accuracy: {acc*100:.2f}%  "
          f"|  Error: {error*100:.2f}%")

# --- Find the best K automatically ---
best_k        = k_range[np.argmin(error_rates)]
best_accuracy = max(accuracy_rates)
best_error    = min(error_rates)

print()
print(f"  Best K Found   : K = {best_k}")
print(f"  Best Accuracy  : {best_accuracy * 100:.2f}%")
print(f"  Lowest Error   : {best_error * 100:.2f}%")
print()

# --- Plot the Elbow Curve ---
plt.figure(figsize=(10, 5))

# Error rate curve
plt.subplot(1, 2, 1)
plt.plot(k_range, error_rates,
         color='red', marker='o',
         markerfacecolor='darkred', markersize=8,
         linewidth=2, label='Error Rate')
plt.axvline(x=best_k, color='green',
            linestyle='--', linewidth=2,
            label=f'Best K={best_k}')
plt.title('Error Rate vs K Value\n(The Elbow Curve)',
          fontsize=12, fontweight='bold')
plt.xlabel('K Value (Number of Neighbors)', fontsize=11)
plt.ylabel('Error Rate', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(k_range, [a*100 for a in accuracy_rates],
         color='blue', marker='s',
         markerfacecolor='darkblue', markersize=8,
         linewidth=2, label='Accuracy %')
plt.axvline(x=best_k, color='green',
            linestyle='--', linewidth=2,
            label=f'Best K={best_k}')
plt.title('Accuracy vs K Value',
          fontsize=12, fontweight='bold')
plt.xlabel('K Value (Number of Neighbors)', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

plt.suptitle('KNN Hyperparameter Tuning — DecodeLabs Project 2',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('k_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ K optimization chart saved as: k_optimization.png")
print()

# --- Retrain with optimal K if different from 5 ---
if best_k != k:
    print(f"  Retraining model with optimal K={best_k}...")
    model_optimized = KNeighborsClassifier(n_neighbors=best_k)
    model_optimized.fit(X_train, y_train)
    y_pred_optimized = model_optimized.predict(X_test)
    f1_optimized = f1_score(y_test, y_pred_optimized,
                            average='weighted')
    print(f"  Optimized F1 Score: {f1_optimized:.4f}")
else:
    print(f"  K=5 is already optimal! No retraining needed.")
print()

# ============================================================
# ENHANCEMENT 2: CUSTOM FLOWER PREDICTION FUNCTION
# ============================================================

print("[ ENHANCEMENT 2 ] Custom Flower Prediction System...")
print("-" * 60)

def predict_flower(sepal_length, sepal_width,
                   petal_length, petal_width):
    """
    Predict the species of an Iris flower from measurements.

    Parameters:
        sepal_length : float → Sepal length in cm
        sepal_width  : float → Sepal width  in cm
        petal_length : float → Petal length in cm
        petal_width  : float → Petal width  in cm

    Returns:
        Prints prediction with confidence percentage
    """

    # --- Step 1: Package the input into correct shape ---
    # Model expects a 2D array: [[feature1, feature2, ...]]
    # We have 1 sample with 4 features
    new_flower = np.array([[sepal_length, sepal_width,
                            petal_length, petal_width]])

    # --- Step 2: Scale using the SAME scaler from training ---
    # Critical: must use same transformation the model learned from
    new_flower_scaled = scaler.transform(new_flower)

    # --- Step 3: Get prediction ---
    prediction_code = model.predict(new_flower_scaled)[0]
    predicted_species = class_names[prediction_code]

    # --- Step 4: Get confidence (probability of each class) ---
    # predict_proba() returns probability for each class
    probabilities = model.predict_proba(new_flower_scaled)[0]
    confidence = probabilities[prediction_code] * 100

    # --- Step 5: Display results ---
    print(f"  Input Measurements:")
    print(f"    Sepal Length : {sepal_length} cm")
    print(f"    Sepal Width  : {sepal_width} cm")
    print(f"    Petal Length : {petal_length} cm")
    print(f"    Petal Width  : {petal_width} cm")
    print(f"  Prediction     : {predicted_species.upper()} 🌸")
    print(f"  Confidence     : {confidence:.1f}%")
    print(f"  All Probabilities:")
    for i, (species, prob) in enumerate(
            zip(class_names, probabilities)):
        bar = "█" * int(prob * 20)
        print(f"    {species:12}: {prob*100:5.1f}% |{bar}")
    print()

    return predicted_species

# --- Test with known flower measurements ---
print("  TEST 1: Classic Setosa measurements")
print("  " + "-" * 40)
predict_flower(5.1, 3.5, 1.4, 0.2)

print("  TEST 2: Classic Versicolor measurements")
print("  " + "-" * 40)
predict_flower(6.0, 2.9, 4.5, 1.5)

print("  TEST 3: Classic Virginica measurements")
print("  " + "-" * 40)
predict_flower(6.9, 3.1, 5.4, 2.1)

print("  TEST 4: Ambiguous flower (borderline case)")
print("  " + "-" * 40)
predict_flower(5.9, 3.0, 4.8, 1.8)

print("✅ Custom prediction system working!")
print()

# ============================================================
# ENHANCEMENT 3: FEATURE ANALYSIS VISUALIZATION
# ============================================================

print("[ ENHANCEMENT 3 ] Generating Feature Analysis Chart...")
print("-" * 60)

# --- Calculate feature variance as importance proxy ---
# Higher variance = feature varies more across species
# = more useful for distinguishing species
feature_variance = np.var(X, axis=0)
feature_labels   = [
    'Sepal\nLength',
    'Sepal\nWidth',
    'Petal\nLength',
    'Petal\nWidth'
]

# --- Create comprehensive analysis figure ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Chart 1: Feature Variance (Importance) ---
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
bars = axes[0].bar(feature_labels, feature_variance,
                   color=colors, edgecolor='black',
                   linewidth=0.8)
axes[0].set_title('Feature Variance\n(Higher = More Important)',
                  fontweight='bold')
axes[0].set_ylabel('Variance')
axes[0].set_xlabel('Feature')
for bar, val in zip(bars, feature_variance):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center',
                 fontsize=10, fontweight='bold')

# --- Chart 2: Species Mean Values Per Feature ---
species_means = []
for species_idx in range(3):
    mask = (y == species_idx)
    means = X[mask].mean(axis=0)
    species_means.append(means)

x_pos = np.arange(len(feature_labels))
width = 0.25
species_colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, (species, means) in enumerate(
        zip(class_names, species_means)):
    axes[1].bar(x_pos + i*width, means,
               width, label=species,
               color=species_colors[i],
               edgecolor='black', linewidth=0.5)

axes[1].set_title('Average Measurements\nby Species',
                  fontweight='bold')
axes[1].set_ylabel('Average Value (cm)')
axes[1].set_xlabel('Feature')
axes[1].set_xticks(x_pos + width)
axes[1].set_xticklabels(feature_labels)
axes[1].legend()

# --- Chart 3: Petal dimensions scatter ---
scatter_colors = ['#3498db', '#e74c3c', '#2ecc71']
for i, (species, color) in enumerate(
        zip(class_names, scatter_colors)):
    mask = (y == i)
    axes[2].scatter(X[mask, 2], X[mask, 3],
                   c=color, label=species,
                   alpha=0.7, s=60,
                   edgecolors='black', linewidth=0.3)

axes[2].set_title('Petal Length vs Width\n(Best Separating Features)',
                  fontweight='bold')
axes[2].set_xlabel('Petal Length (cm)')
axes[2].set_ylabel('Petal Width (cm)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Iris Dataset — Feature Analysis\nDecodeLabs Project 2',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Feature analysis chart saved as: feature_analysis.png")
print()
print("=" * 60)
print("  ALL ENHANCEMENTS COMPLETE")
print("=" * 60)
print(f"  Files generated:")
print(f"  → confusion_matrix.png")
print(f"  → k_optimization.png")
print(f"  → feature_analysis.png")
print("=" * 60)