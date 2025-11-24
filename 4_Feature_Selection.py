import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, recall_score, roc_curve, auc, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest, f_classif
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import warnings
import gc
from tqdm import tqdm
import torch.nn.functional as F

# Feature Selection Libraries
import shap
from mrmr import mrmr_classif
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier as RF_Boruta
from deap import base, creator, tools, algorithms
import seaborn as sns

warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style("whitegrid")

# Base directory
BASE_DIR = "4_Feature_Selection_Comparison/SwinTransformer"
os.makedirs(BASE_DIR, exist_ok=True)

# Device setup
device = torch.device("cpu")
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device detected, using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device detected, using CUDA")
    print(f"Using device: {device}")
except:
    print("Error detecting device capabilities, defaulting to CPU")


# Load and prepare data
try:
    print("Loading and preparing data...")
    data = pd.read_csv('2_Feature_Extraction/SwinTransformer/feature_extract_SwinTransformer_features.csv')

    # Extract features (columns 0-1023, which are the first 1024 columns)
    X = data.iloc[:, :1024].values  # First 1024 columns are features
    y = data.iloc[:, 1024].values  # Column 1024 is the label

    num_classes = len(np.unique(y))
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes")
    print(f"Feature columns: 0-{X.shape[1] - 1}")
    print(f"Label column: {X.shape[1]} (values: {np.unique(y)})")
    print("Class distribution:")
    print(pd.Series(y).value_counts(normalize=True))

    # Verify we have exactly 1024 features
    assert X.shape[1] == 1024, f"Expected 1024 features, got {X.shape[1]}"

except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")



# Multi-Scale CNN Model
class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes, input_dim=500):
        super(MultiScaleCNN, self).__init__()

        self.input_dim = input_dim

        # Branch 1: Fine-grained features
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 160, kernel_size=3, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Branch 2: Medium-scale features
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 160, kernel_size=5, padding=2),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Branch 3: Large-scale features
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=7, padding=3),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 160, kernel_size=7, padding=3),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Global context branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(1, 160, kernel_size=1),
            nn.BatchNorm1d(160),
            nn.ReLU()
        )

        # Calculate feature map size after pooling
        self.feature_size = input_dim // 4  # After 2 pooling operations

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=160, num_heads=8, batch_first=True)

        # Fusion layers
        self.fusion1 = nn.Sequential(
            nn.Conv1d(160 * 3 + 160, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fusion2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, 1, features]

        # Multi-scale feature extraction
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)

        # Global context
        global_feat = self.global_branch(x)
        global_feat = global_feat.expand(-1, -1, self.feature_size)

        # Apply attention to first branch
        feat1_att = feat1.transpose(1, 2)
        feat1_att, _ = self.attention(feat1_att, feat1_att, feat1_att)
        feat1_att = feat1_att.transpose(1, 2)

        # Concatenate features
        combined = torch.cat([feat1_att, feat2, feat3, global_feat], dim=1)

        # Fusion and classification
        fused1 = self.fusion1(combined)
        fused2 = self.fusion2(fused1)
        output = self.classifier(fused2)

        return output



# Feature Selection Techniques

def shap_feature_selection(X_train, y_train, n_features=500):
    """SHAP-based feature selection with explainability"""
    print("Applying SHAP feature selection...")

    # Train a simple model for SHAP analysis
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)

    # Use a subset of training data for SHAP if dataset is small
    max_samples = min(100, X_train.shape[0])
    sample_indices = np.random.choice(X_train.shape[0], max_samples, replace=False)
    X_shap_sample = X_train[sample_indices]

    print(f"Computing SHAP values for {max_samples} samples...")
    shap_values = explainer.shap_values(X_shap_sample)

    print(f"SHAP values shape: {np.array(shap_values).shape if isinstance(shap_values, list) else shap_values.shape}")

    # Handle different SHAP output formats for binary classification
    if isinstance(shap_values, list):
        if len(shap_values) == 2:  # Binary classification with 2 classes
            # Use the positive class (class 1) SHAP values
            shap_vals_for_analysis = shap_values[1]  # Shape: (samples, features)
            shap_vals_for_plot = shap_values[1]
        else:  # Multi-class
            # Average across all classes
            shap_vals_for_analysis = np.mean(shap_values, axis=0)
            shap_vals_for_plot = shap_values[0]
    else:
        # Single array output - shape should be (samples, features)
        shap_vals_for_analysis = shap_values
        shap_vals_for_plot = shap_values

    print(f"SHAP values for analysis shape: {shap_vals_for_analysis.shape}")

    # Ensure we have 2D array: (samples, features)
    if len(shap_vals_for_analysis.shape) > 2:
        print(f"Reshaping SHAP values from {shap_vals_for_analysis.shape}")
        # For binary classification, often we get (samples, features, 2)
        # Take the absolute max across the last dimension or use class 1
        if shap_vals_for_analysis.shape[-1] == 2:
            # Use class 1 (positive class) values
            shap_vals_for_analysis = shap_vals_for_analysis[:, :, 1]
        else:
            # Flatten extra dimensions
            shap_vals_for_analysis = shap_vals_for_analysis.reshape(shap_vals_for_analysis.shape[0], -1)

    print(f"Final SHAP values shape for analysis: {shap_vals_for_analysis.shape}")

    # Calculate mean absolute SHAP values across samples
    if len(shap_vals_for_analysis.shape) == 2:
        mean_shap_values = np.abs(shap_vals_for_analysis).mean(axis=0)
    else:
        print("Warning: Unexpected SHAP values shape, using Random Forest feature importances")
        mean_shap_values = rf_model.feature_importances_

    print(f"Mean SHAP values shape: {mean_shap_values.shape}")

    # Ensure we have the right number of features
    if len(mean_shap_values) != X_train.shape[1]:
        print(f"Warning: SHAP values length ({len(mean_shap_values)}) doesn't match features ({X_train.shape[1]})")
        # Fallback to feature importance from random forest
        mean_shap_values = rf_model.feature_importances_
        print("Using Random Forest feature importances instead")

    # Select top features based on mean absolute SHAP values
    selected_indices = np.argsort(mean_shap_values)[-n_features:]

    # Generate plots for top 20 features
    top_20_indices = np.argsort(mean_shap_values)[-20:]
    top_20_names = [f'Feature_{i}' for i in top_20_indices]
    top_20_importance = mean_shap_values[top_20_indices]

    print(f"Top 20 importance values shape: {top_20_importance.shape}")
    print(f"Sample values: {top_20_importance[:3]}")

    # Create SHAP summary plot (try with error handling)
    try:
        plt.figure(figsize=(12, 8))

        # Prepare data for SHAP plot - ensure 2D
        plot_shap_values = shap_vals_for_plot
        if len(plot_shap_values.shape) > 2:
            if plot_shap_values.shape[-1] == 2:
                plot_shap_values = plot_shap_values[:, :, 1]  # Use positive class
            else:
                plot_shap_values = plot_shap_values.reshape(plot_shap_values.shape[0], -1)

        # Only use if shapes are compatible
        if plot_shap_values.shape[1] == X_shap_sample.shape[1]:
            shap.summary_plot(plot_shap_values[:, top_20_indices],
                              X_shap_sample[:, top_20_indices],
                              feature_names=top_20_names,
                              show=False,
                              max_display=20)

            plt.title('SHAP Summary Plot - Top 20 Features', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, 'shap_summary_plot.png'), dpi=1000, bbox_inches='tight')
            plt.savefig(os.path.join(BASE_DIR, 'shap_summary_plot.pdf'), dpi=1000, bbox_inches='tight')
            plt.close()
            print("SHAP summary plot created successfully")
        else:
            raise ValueError(f"Shape mismatch: SHAP {plot_shap_values.shape} vs Features {X_shap_sample.shape}")

    except Exception as e:
        print(f"Warning: Could not create SHAP summary plot: {e}")
        print("Creating alternative feature importance plot instead...")

    # Feature importance bar plot (always create this as backup)
    try:
        plt.figure(figsize=(12, 8))

        # Ensure top_20_importance is 1D
        if len(top_20_importance.shape) > 1:
            top_20_importance = top_20_importance.flatten()

        plt.barh(range(len(top_20_names)), top_20_importance)
        plt.yticks(range(len(top_20_names)), top_20_names)
        plt.xlabel('Mean |SHAP Value|', fontsize=14)
        plt.title('Top 20 Features by SHAP Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.gca().grid(False)
        plt.tight_layout()
        plt.grid(False)
        plt.savefig(os.path.join(BASE_DIR, 'shap_feature_importance_bar.png'), dpi=1000, bbox_inches='tight')
        plt.savefig(os.path.join(BASE_DIR, 'shap_feature_importance_bar.pdf'), dpi=1000, bbox_inches='tight')
        plt.close()
        print("SHAP feature importance bar plot created successfully")

    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
        print(f"top_20_importance shape: {top_20_importance.shape}")
        print(f"top_20_names length: {len(top_20_names)}")

    print(f"SHAP: Selected {len(selected_indices)} features")
    return selected_indices, mean_shap_values


def mrmr_feature_selection(X_train, y_train, n_features=500):
    """mRMR feature selection"""
    print("Applying mRMR feature selection...")

    # Convert to DataFrame for mRMR
    df_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    df_train['target'] = y_train

    # Apply mRMR
    selected_features = mrmr_classif(X=df_train.drop('target', axis=1),
                                     y=df_train['target'],
                                     K=n_features)

    # Convert feature names back to indices
    selected_indices = [int(feat.split('_')[1]) for feat in selected_features]

    print(f"mRMR: Selected {len(selected_indices)} features")
    return selected_indices


def lasso_feature_selection(X_train, y_train, n_features=500):
    """Lasso (L1) regularization feature selection"""
    print("Applying Lasso feature selection...")

    # Use LassoCV for automatic alpha selection
    lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
    lasso.fit(X_train, y_train)

    # Get feature coefficients
    coefficients = np.abs(lasso.coef_)

    # Select top features
    selected_indices = np.argsort(coefficients)[-n_features:]

    print(f"Lasso: Selected {len(selected_indices)} features")
    return selected_indices, coefficients


def boruta_feature_selection(X_train, y_train, n_features=500):
    """Boruta all-relevant feature selection"""
    print("Applying Boruta feature selection...")

    # Initialize Boruta
    rf_boruta = RF_Boruta(n_estimators=100, random_state=42, n_jobs=-1)
    boruta_selector = BorutaPy(rf_boruta, n_estimators='auto', random_state=42, max_iter=50)

    # Fit Boruta
    boruta_selector.fit(X_train, y_train)

    # Get selected features
    selected_mask = boruta_selector.support_
    selected_indices = np.where(selected_mask)[0]

    # If more than n_features selected, take top by ranking
    if len(selected_indices) > n_features:
        rankings = boruta_selector.ranking_[selected_indices]
        top_indices = np.argsort(rankings)[:n_features]
        selected_indices = selected_indices[top_indices]
    elif len(selected_indices) < n_features:
        # If fewer features selected, add next best features
        unselected_mask = ~selected_mask
        unselected_indices = np.where(unselected_mask)[0]
        unselected_rankings = boruta_selector.ranking_[unselected_indices]
        additional_needed = n_features - len(selected_indices)
        additional_indices = np.argsort(unselected_rankings)[:additional_needed]
        selected_indices = np.concatenate([selected_indices, unselected_indices[additional_indices]])

    print(f"Boruta: Selected {len(selected_indices)} features")
    return selected_indices, boruta_selector.ranking_


def genetic_algorithm_feature_selection(X_train, y_train, n_features=500, population_size=30, generations=15):
    """Genetic Algorithm feature selection"""
    print("Applying Genetic Algorithm feature selection...")

    # Define fitness function
    def evaluate_features(individual):
        selected_features = [i for i, x in enumerate(individual) if x == 1]
        if len(selected_features) == 0:
            return 0,

        # Quick evaluation using subset of data
        subset_size = min(200, X_train.shape[0])
        indices = np.random.choice(X_train.shape[0], subset_size, replace=False)
        X_subset = X_train[indices][:, selected_features]
        y_subset = y_train[indices]

        # Simple RF evaluation
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_subset, y_subset)
        score = rf.score(X_subset, y_subset)

        # Penalty for too many or too few features
        feature_ratio = len(selected_features) / X_train.shape[1]
        if feature_ratio < 0.3:  # Too few features
            penalty = (0.3 - feature_ratio) * 0.5
        elif feature_ratio > 0.7:  # Too many features
            penalty = (feature_ratio - 0.7) * 0.5
        else:
            penalty = 0

        fitness = score - penalty
        return fitness,

    # Setup DEAP
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.choice, [0, 1], p=[0.5, 0.5])
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, X_train.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_features)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    population = toolbox.population(n=population_size)

    # Evolution
    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")

        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Selection and breeding
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population = offspring

    # Get best individual
    best_individual = tools.selBest(population, 1)[0]
    selected_indices = [i for i, x in enumerate(best_individual) if x == 1]

    # Ensure we have exactly n_features
    if len(selected_indices) > n_features:
        selected_indices = np.random.choice(selected_indices, n_features, replace=False)
    elif len(selected_indices) < n_features:
        remaining = [i for i in range(X_train.shape[1]) if i not in selected_indices]
        additional = np.random.choice(remaining, n_features - len(selected_indices), replace=False)
        selected_indices = np.concatenate([selected_indices, additional])

    print(f"Genetic Algorithm: Selected {len(selected_indices)} features")
    return selected_indices



# Training and evaluation functions
def create_data_loaders(X_train_subset, y_train_subset, X_val, y_val, batch_size=32):
    X_train_3d = torch.FloatTensor(X_train_subset).unsqueeze(-1)
    X_val_3d = torch.FloatTensor(X_val).unsqueeze(-1)
    X_test_3d = torch.FloatTensor(X_test).unsqueeze(-1)
    y_train_t = torch.LongTensor(y_train_subset)
    y_val_t = torch.LongTensor(y_val)
    y_test_t = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_3d, y_train_t)
    val_dataset = TensorDataset(X_val_3d, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, X_test_3d, y_test_t


def train_model(model, train_loader, val_loader, epochs=25):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        history['loss'].append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()

        val_loss = val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {history['loss'][-1]:.4f}, Val Loss: {val_loss:.4f}")

    return model, history


def evaluate_model(model, X_test_selected, y_test):
    model.eval()
    X_test_3d = torch.FloatTensor(X_test_selected).unsqueeze(-1).to(device)

    with torch.no_grad():
        y_pred = model(X_test_3d).cpu().numpy()

    y_pred_prob = torch.softmax(torch.tensor(y_pred), dim=1).numpy()
    y_pred_binary = np.argmax(y_pred_prob, axis=1)

    # Calculate confusion matrix for SP and SN
    cm = confusion_matrix(y_test, y_pred_binary)

    # For binary classification
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        # Sensitivity (SN) = Recall = TPR = TP/(TP+FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Specificity (SP) = TNR = TN/(TN+FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # For multiclass, use macro average
        sensitivity = recall_score(y_test, y_pred_binary, average='macro')
        specificity = recall_score(y_test, y_pred_binary, average='macro')  # Same as recall for multiclass

    results = {
        'ACC': accuracy_score(y_test, y_pred_binary),
        'PRE': precision_score(y_test, y_pred_binary, average='macro'),
        'SP': specificity,
        'SN': sensitivity,
        'F1': f1_score(y_test, y_pred_binary, average='macro'),
        'MCC': matthews_corrcoef(y_test, y_pred_binary)
    }

    # Calculate AUC and store ROC curve data
    auc_scores = []
    roc_curves = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        roc_curves.append((fpr, tpr, auc_score))

    results['AUC'] = np.mean(auc_scores)

    return results, roc_curves



# Main feature selection comparison
def run_feature_selection_comparison():
    print("=" * 80)
    print("FEATURE SELECTION TECHNIQUES COMPARISON")
    print("=" * 80)

    # Store results
    all_results = {}
    selected_features = {}
    all_roc_data = {}

    # 1. SHAP Feature Selection
    print("\n1. SHAP Feature Selection")
    print("-" * 40)
    start_time = time.time()
    shap_indices, shap_values = shap_feature_selection(X_train, y_train, n_features=500)
    selected_features['SHAP'] = shap_indices
    shap_time = time.time() - start_time

    # Train and evaluate with SHAP features
    X_train_shap = X_train[:, shap_indices]
    X_val_shap = X_val[:, shap_indices]
    X_test_shap = X_test[:, shap_indices]

    model_shap = MultiScaleCNN(num_classes, input_dim=500)
    train_loader, val_loader, _, _ = create_data_loaders(X_train_shap, y_train, X_val_shap, y_val)
    model_shap, _ = train_model(model_shap, train_loader, val_loader)
    results_shap, roc_shap = evaluate_model(model_shap, X_test_shap, y_test)
    results_shap['Selection_Time'] = shap_time
    all_results['SHAP'] = results_shap
    all_roc_data['SHAP'] = roc_shap

    # Clear memory
    del model_shap, train_loader, val_loader
    gc.collect()

    # 2. mRMR Feature Selection
    print("\n2. mRMR Feature Selection")
    print("-" * 40)
    start_time = time.time()
    mrmr_indices = mrmr_feature_selection(X_train, y_train, n_features=500)
    selected_features['mRMR'] = mrmr_indices
    mrmr_time = time.time() - start_time

    X_train_mrmr = X_train[:, mrmr_indices]
    X_val_mrmr = X_val[:, mrmr_indices]
    X_test_mrmr = X_test[:, mrmr_indices]

    model_mrmr = MultiScaleCNN(num_classes, input_dim=500)
    train_loader, val_loader, _, _ = create_data_loaders(X_train_mrmr, y_train, X_val_mrmr, y_val)
    model_mrmr, _ = train_model(model_mrmr, train_loader, val_loader)
    results_mrmr, roc_mrmr = evaluate_model(model_mrmr, X_test_mrmr, y_test)
    results_mrmr['Selection_Time'] = mrmr_time
    all_results['mRMR'] = results_mrmr
    all_roc_data['mRMR'] = roc_mrmr

    # Clear memory
    del model_mrmr, train_loader, val_loader
    gc.collect()

    # 3. Lasso Feature Selection
    print("\n3. Lasso (L1) Feature Selection")
    print("-" * 40)
    start_time = time.time()
    lasso_indices, lasso_coef = lasso_feature_selection(X_train, y_train, n_features=500)
    selected_features['Lasso'] = lasso_indices
    lasso_time = time.time() - start_time

    X_train_lasso = X_train[:, lasso_indices]
    X_val_lasso = X_val[:, lasso_indices]
    X_test_lasso = X_test[:, lasso_indices]

    model_lasso = MultiScaleCNN(num_classes, input_dim=500)
    train_loader, val_loader, _, _ = create_data_loaders(X_train_lasso, y_train, X_val_lasso, y_val)
    model_lasso, _ = train_model(model_lasso, train_loader, val_loader)
    results_lasso, roc_lasso = evaluate_model(model_lasso, X_test_lasso, y_test)
    results_lasso['Selection_Time'] = lasso_time
    all_results['Lasso'] = results_lasso
    all_roc_data['Lasso'] = roc_lasso

    # Clear memory
    del model_lasso, train_loader, val_loader
    gc.collect()

    # 4. Boruta Feature Selection
    print("\n4. Boruta Feature Selection")
    print("-" * 40)
    start_time = time.time()
    boruta_indices, boruta_ranking = boruta_feature_selection(X_train, y_train, n_features=500)
    selected_features['Boruta'] = boruta_indices
    boruta_time = time.time() - start_time

    X_train_boruta = X_train[:, boruta_indices]
    X_val_boruta = X_val[:, boruta_indices]
    X_test_boruta = X_test[:, boruta_indices]

    model_boruta = MultiScaleCNN(num_classes, input_dim=500)
    train_loader, val_loader, _, _ = create_data_loaders(X_train_boruta, y_train, X_val_boruta, y_val)
    model_boruta, _ = train_model(model_boruta, train_loader, val_loader)
    results_boruta, roc_boruta = evaluate_model(model_boruta, X_test_boruta, y_test)
    results_boruta['Selection_Time'] = boruta_time
    all_results['Boruta'] = results_boruta
    all_roc_data['Boruta'] = roc_boruta

    # Clear memory
    del model_boruta, train_loader, val_loader
    gc.collect()

    # 5. Genetic Algorithm Feature Selection
    print("\n5. Genetic Algorithm Feature Selection")
    print("-" * 40)
    start_time = time.time()
    ga_indices = genetic_algorithm_feature_selection(X_train, y_train, n_features=500)
    selected_features['Genetic_Algorithm'] = ga_indices
    ga_time = time.time() - start_time

    X_train_ga = X_train[:, ga_indices]
    X_val_ga = X_val[:, ga_indices]
    X_test_ga = X_test[:, ga_indices]

    model_ga = MultiScaleCNN(num_classes, input_dim=500)
    train_loader, val_loader, _, _ = create_data_loaders(X_train_ga, y_train, X_val_ga, y_val)
    model_ga, _ = train_model(model_ga, train_loader, val_loader)
    results_ga, roc_ga = evaluate_model(model_ga, X_test_ga, y_test)
    results_ga['Selection_Time'] = ga_time
    all_results['Genetic_Algorithm'] = results_ga
    all_roc_data['Genetic_Algorithm'] = roc_ga

    # Clear memory
    del model_ga, train_loader, val_loader
    gc.collect()

    return all_results, selected_features, all_roc_data



# Visualization functions
def create_comparison_plots(results):
    """Create comprehensive comparison plots"""

    # Extract metrics
    methods = list(results.keys())
    metrics = ['ACC', 'AUC', 'PRE', 'SP', 'SN', 'F1', 'MCC']

    # Create comparison bar plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [results[method][metric] for method in methods]
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        bars = axes[i].bar(methods, values, color=colors)
        axes[i].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    # Hide the empty subplot
    axes[7].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'feature_selection_comparison.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'feature_selection_comparison.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()

    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        values = [results[method][metric] for metric in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Feature Selection Methods - Performance Radar Chart',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'feature_selection_radar_chart.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'feature_selection_radar_chart.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()

    # Selection time comparison
    plt.figure(figsize=(12, 6))
    selection_times = [results[method]['Selection_Time'] for method in methods]
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

    bars = plt.bar(methods, selection_times, color=colors)
    plt.title('Feature Selection Time Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, time_val in zip(bars, selection_times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(selection_times) * 0.01,
                 f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'selection_time_comparison.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'selection_time_comparison.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()


def create_roc_curves(all_roc_data):
    """Create ROC curves for all feature selection methods"""

    plt.figure(figsize=(12, 10))
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i, (method_name, roc_data) in enumerate(all_roc_data.items()):
        if roc_data and len(roc_data) > 0:
            # For binary classification, use the positive class (class 1)
            # For multiclass, use the first class or average
            if len(roc_data) >= 2:  # Binary classification
                fpr, tpr, auc_score = roc_data[1]  # Use positive class
            else:  # Single class or multiclass
                fpr, tpr, auc_score = roc_data[0]

            plt.plot(fpr, tpr, color=colors[i % len(colors)],
                     label=f'{method_name} (AUC = {auc_score:.4f})',
                     linewidth=3)

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=16)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=16)
    plt.title('ROC Curves - Feature Selection Methods Comparison', fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Style the plot
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'roc_curves_comparison.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'roc_curves_comparison.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()
    plt.close()

    print("ROC curves comparison plot created successfully")


def analyze_feature_overlap(selected_features):
    """Analyze overlap between different feature selection methods"""

    methods = list(selected_features.keys())

    # Create overlap matrix
    overlap_matrix = np.zeros((len(methods), len(methods)))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            set1 = set(selected_features[method1])
            set2 = set(selected_features[method2])
            overlap = len(set1.intersection(set2))
            overlap_matrix[i, j] = overlap / len(set1.union(set2))  # Jaccard similarity

    # Plot overlap heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(overlap_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=methods, yticklabels=methods, cbar_kws={'label': 'Jaccard Similarity'})
    plt.title('Feature Selection Methods - Feature Overlap Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'feature_overlap_heatmap.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'feature_overlap_heatmap.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()

    return overlap_matrix


def create_feature_distribution_plots(selected_features):
    """Create plots showing feature distribution patterns"""

    methods = list(selected_features.keys())

    # Feature index distribution
    plt.figure(figsize=(15, 10))

    for i, method in enumerate(methods):
        plt.subplot(2, 3, i + 1)
        feature_indices = selected_features[method]
        plt.hist(feature_indices, bins=50, alpha=0.7, color=plt.cm.Set3(i / len(methods)))
        plt.title(f'{method}\nFeature Index Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Feature Index')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'feature_distribution_plots.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'feature_distribution_plots.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()



# Execute the comparison
if __name__ == "__main__":
    """
    Main function to execute the complete feature selection comparison pipeline
    """
    print("Starting Feature Selection Comparison...")
    print("=" * 80)
    print("FEATURE SELECTION PIPELINE INITIATED")
    print("=" * 80)

    try:
        # Run comparison
        print("🚀 Running feature selection comparison...")
        results, selected_features, all_roc_data = run_feature_selection_comparison()

        # Create results summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        results_df = pd.DataFrame(results).T
        print(results_df.round(4))

        # Save results
        print("\n💾 Saving results...")
        results_df.to_csv(os.path.join(BASE_DIR, 'feature_selection_results.csv'))
        print(f"✅ Results saved to: {os.path.join(BASE_DIR, 'feature_selection_results.csv')}")

        # Create visualizations
        print("\n📊 Creating comparison plots...")
        create_comparison_plots(results)

        print("\n📈 Creating ROC curves...")
        create_roc_curves(all_roc_data)

        print("\n🔍 Analyzing feature overlap...")
        overlap_matrix = analyze_feature_overlap(selected_features)

        print("\n📋 Creating feature distribution plots...")
        create_feature_distribution_plots(selected_features)

        # Save selected features
        print("\n💾 Saving selected features...")
        for method, features in selected_features.items():
            feature_file = os.path.join(BASE_DIR, f'{method}_selected_features.npy')
            np.save(feature_file, features)
            print(f"✅ {method} features saved to: {feature_file}")

        # Performance ranking
        print("\n" + "=" * 80)
        print("PERFORMANCE RANKING BY ACCURACY")
        print("=" * 80)

        accuracy_ranking = sorted(results.items(), key=lambda x: x[1]['ACC'], reverse=True)
        for i, (method, result) in enumerate(accuracy_ranking, 1):
            print(f"{i}. {method}: {result['ACC']:.4f}")

        # Key insights
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)

        best_method = accuracy_ranking[0][0]
        best_acc = accuracy_ranking[0][1]['ACC']
        print(f"🏆 Best performing method: {best_method} ({best_acc:.4f} accuracy)")

        # Find fastest method
        fastest_method = min(results.items(), key=lambda x: x[1]['Selection_Time'])
        print(f"⚡ Fastest selection method: {fastest_method[0]} ({fastest_method[1]['Selection_Time']:.2f}s)")

        # Find most stable method (highest MCC)
        most_stable = max(results.items(), key=lambda x: x[1]['MCC'])
        print(f"📊 Most stable method (highest MCC): {most_stable[0]} ({most_stable[1]['MCC']:.4f})")

        # Find best AUC
        best_auc = max(results.items(), key=lambda x: x[1]['AUC'])
        print(f"🎯 Best AUC method: {best_auc[0]} ({best_auc[1]['AUC']:.4f})")

        # Find best specificity
        best_sp = max(results.items(), key=lambda x: x[1]['SP'])
        print(f"🎯 Best Specificity method: {best_sp[0]} ({best_sp[1]['SP']:.4f})")

        # Find best sensitivity
        best_sn = max(results.items(), key=lambda x: x[1]['SN'])
        print(f"🎯 Best Sensitivity method: {best_sn[0]} ({best_sn[1]['SN']:.4f})")

        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        print("📊 Metric Averages Across All Methods:")
        for metric in ['ACC', 'AUC', 'PRE', 'SP', 'SN', 'F1', 'MCC']:
            avg_value = np.mean([results[method][metric] for method in results.keys()])
            std_value = np.std([results[method][metric] for method in results.keys()])
            print(f"   {metric}: {avg_value:.4f} ± {std_value:.4f}")

        # Final completion message
        print("\n" + "=" * 80)
        print("COMPLETION STATUS")
        print("=" * 80)

        print("✅ Feature selection comparison completed successfully!")
        print(f"📁 All results saved to: {BASE_DIR}")
        print(f"📊 SHAP summary plot generated for top 20 features")
        print(f"📈 ROC curves created for all methods")
        print(f"🔬 {len(selected_features)} feature selection methods compared")
        print(f"🎯 500 features selected from 1024 original features by each method")
        print(f"📋 {len(results)} performance metrics calculated per method")

        # File summary
        print("\n📁 Generated Files:")
        generated_files = [
            'feature_selection_results.csv',
            'feature_selection_comparison.png',
            'feature_selection_radar_chart.png',
            'roc_curves_comparison.png',
            'shap_summary_plot.png',
            'feature_overlap_heatmap.png',
            'feature_distribution_plots.png',
            'selection_time_comparison.png'
        ]

        for file in generated_files:
            file_path = os.path.join(BASE_DIR, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (not found)")

        # Feature files
        print("\n📋 Feature Selection Files:")
        for method in selected_features.keys():
            feature_file = f'{method}_selected_features.npy'
            file_path = os.path.join(BASE_DIR, feature_file)
            if os.path.exists(file_path):
                print(f"   ✅ {feature_file}")
            else:
                print(f"   ❌ {feature_file} (not found)")

    except Exception as e:
        print(f"❌ Error occurred during execution: {e}")
        print("Please check the error details above and ensure all dependencies are installed.")
        raise

    finally:
        # Memory cleanup
        print("\n🧹 Performing memory cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("✅ Memory cleanup completed")

        print("\n🎉 Analysis complete!")
        print("=" * 80)
