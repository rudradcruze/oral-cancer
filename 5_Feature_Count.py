import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, recall_score, roc_curve, auc, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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
import shap
import seaborn as sns

warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style("whitegrid")

# Base directory
BASE_DIR = "5_Feature_Count/SwinTransformer"
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


# Multi-Scale CNN Model (Dynamic input dimension)
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


# SHAP Feature Selection Function
def shap_feature_selection(X_train, y_train, n_features=500):
    """SHAP-based feature selection with explainability"""
    print(f"Applying SHAP feature selection for {n_features} features...")

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

    # Handle different SHAP output formats for binary classification
    if isinstance(shap_values, list):
        if len(shap_values) == 2:  # Binary classification with 2 classes
            shap_vals_for_analysis = shap_values[1]  # Use positive class
            shap_vals_for_plot = shap_values[1]
        else:  # Multi-class
            shap_vals_for_analysis = np.mean(shap_values, axis=0)
            shap_vals_for_plot = shap_values[0]
    else:
        shap_vals_for_analysis = shap_values
        shap_vals_for_plot = shap_values

    # Ensure we have 2D array: (samples, features)
    if len(shap_vals_for_analysis.shape) > 2:
        if shap_vals_for_analysis.shape[-1] == 2:
            shap_vals_for_analysis = shap_vals_for_analysis[:, :, 1]
        else:
            shap_vals_for_analysis = shap_vals_for_analysis.reshape(shap_vals_for_analysis.shape[0], -1)

    # Calculate mean absolute SHAP values across samples
    if len(shap_vals_for_analysis.shape) == 2:
        mean_shap_values = np.abs(shap_vals_for_analysis).mean(axis=0)
    else:
        print("Warning: Using Random Forest feature importances")
        mean_shap_values = rf_model.feature_importances_

    # Ensure we have the right number of features
    if len(mean_shap_values) != X_train.shape[1]:
        mean_shap_values = rf_model.feature_importances_

    # Select top features based on mean absolute SHAP values
    selected_indices = np.argsort(mean_shap_values)[-n_features:]

    # Create feature importance plot for each feature count
    if n_features <= 500:  # Only create plots for reasonable numbers
        top_20_indices = np.argsort(mean_shap_values)[-min(20, n_features):]
        top_20_names = [f'Feature_{i}' for i in top_20_indices]
        top_20_importance = mean_shap_values[top_20_indices]

        try:
            plt.figure(figsize=(12, 8))
            if len(top_20_importance.shape) > 1:
                top_20_importance = top_20_importance.flatten()

            plt.barh(range(len(top_20_names)), top_20_importance)
            plt.yticks(range(len(top_20_names)), top_20_names)
            plt.xlabel('Mean |SHAP Value|', fontsize=14)
            plt.title(f'Top {len(top_20_names)} Features by SHAP Importance ({n_features} total selected)',
                      fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.gca().grid(False)
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, f'shap_importance_{n_features}_features.png'),
                        dpi=1000, bbox_inches='tight')
            plt.savefig(os.path.join(BASE_DIR, f'shap_importance_{n_features}_features.pdf'),
                        dpi=1000, bbox_inches='tight')
            plt.close()
            print(f"SHAP feature importance plot created for {n_features} features")

        except Exception as e:
            print(f"Error creating feature importance plot for {n_features} features: {e}")

    print(f"SHAP: Selected {len(selected_indices)} features")
    return selected_indices, mean_shap_values


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
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = recall_score(y_test, y_pred_binary, average='macro')
        specificity = recall_score(y_test, y_pred_binary, average='macro')

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


# Main feature count comparison function
def run_feature_count_comparison():
    print("=" * 80)
    print("SHAP FEATURE COUNT COMPARISON")
    print("=" * 80)

    # Feature counts to compare
    feature_counts = [200, 300, 500, 700, 900]

    # Store results
    all_results = {}
    selected_features = {}
    all_roc_data = {}
    shap_values_dict = {}

    # Get SHAP values once (using maximum feature count to ensure we can select any subset)
    print("Computing SHAP values...")
    shap_indices_900, shap_values = shap_feature_selection(X_train, y_train, n_features=900)

    # For each feature count
    for n_features in feature_counts:
        print(f"\n{'=' * 40}")
        print(f"TESTING WITH {n_features} FEATURES")
        print(f"{'=' * 40}")

        start_time = time.time()

        # Select top n_features from the pre-computed SHAP values
        selected_indices = np.argsort(shap_values)[-n_features:]
        selected_features[f'SHAP_{n_features}'] = selected_indices

        # Create feature importance plot
        top_display = min(20, n_features)
        top_indices = np.argsort(shap_values)[-top_display:]
        top_names = [f'Feature_{i}' for i in top_indices]
        top_importance = shap_values[top_indices]

        try:
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_names)), top_importance)
            plt.yticks(range(len(top_names)), top_names)
            plt.xlabel('Mean |SHAP Value|', fontsize=14)
            plt.title(f'Top {top_display} Features by SHAP Importance ({n_features} total selected)',
                      fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.gca().grid(False)
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, f'shap_importance_{n_features}_features.png'),
                        dpi=1000, bbox_inches='tight')
            plt.savefig(os.path.join(BASE_DIR, f'shap_importance_{n_features}_features.pdf'),
                        dpi=1000, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating plot for {n_features} features: {e}")

        selection_time = time.time() - start_time

        # Prepare data with selected features
        X_train_selected = X_train[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]

        print(f"Training with {n_features} features...")

        # Train model
        model = MultiScaleCNN(num_classes, input_dim=n_features)
        train_loader, val_loader, _, _ = create_data_loaders(X_train_selected, y_train, X_val_selected, y_val)

        training_start = time.time()
        model, _ = train_model(model, train_loader, val_loader, epochs=25)
        training_time = time.time() - training_start

        # Evaluate model
        results, roc_curves = evaluate_model(model, X_test_selected, y_test)
        results['Selection_Time'] = selection_time
        results['Training_Time'] = training_time
        results['Total_Time'] = selection_time + training_time
        results['Feature_Count'] = n_features

        # Store results
        all_results[f'SHAP_{n_features}'] = results
        all_roc_data[f'SHAP_{n_features}'] = roc_curves

        print(f"Results for {n_features} features:")
        print(f"  Accuracy: {results['ACC']:.4f}")
        print(f"  AUC: {results['AUC']:.4f}")
        print(f"  F1-Score: {results['F1']:.4f}")
        print(f"  MCC: {results['MCC']:.4f}")
        print(f"  Training Time: {training_time:.2f}s")

        # Clear memory
        del model, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return all_results, selected_features, all_roc_data


# Visualization functions
def create_feature_count_comparison_plots(results):
    """Create comprehensive comparison plots for different feature counts"""

    # Extract data
    feature_counts = [200, 300, 500, 700, 900]
    metrics = ['ACC', 'AUC', 'PRE', 'SP', 'SN', 'F1', 'MCC']

    # Performance vs Feature Count
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [results[f'SHAP_{n}'][metric] for n in feature_counts]

        axes[i].plot(feature_counts, values, 'o-', linewidth=3, markersize=8, color='blue')
        axes[i].set_title(f'{metric} vs Feature Count', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Number of Features', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(feature_counts)

        # Add value labels
        for j, (fc, val) in enumerate(zip(feature_counts, values)):
            axes[i].annotate(f'{val:.3f}', (fc, val), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=10)

    # Hide the empty subplot
    axes[7].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'feature_count_performance_comparison.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'feature_count_performance_comparison.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()

    # Training Time vs Feature Count
    plt.figure(figsize=(12, 8))
    training_times = [results[f'SHAP_{n}']['Training_Time'] for n in feature_counts]

    plt.plot(feature_counts, training_times, 'o-', linewidth=3, markersize=10, color='red', label='Training Time')
    plt.title('Training Time vs Feature Count', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Features', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(feature_counts)

    # Add value labels
    for fc, time_val in zip(feature_counts, training_times):
        plt.annotate(f'{time_val:.1f}s', (fc, time_val), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_time_vs_feature_count.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'training_time_vs_feature_count.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()

    # Combined Performance Overview
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Left plot: Main metrics
    main_metrics = ['ACC', 'AUC', 'F1', 'MCC']
    colors = ['blue', 'green', 'red', 'purple']

    for i, metric in enumerate(main_metrics):
        values = [results[f'SHAP_{n}'][metric] for n in feature_counts]
        ax1.plot(feature_counts, values, 'o-', linewidth=3, markersize=8,
                 color=colors[i], label=metric)

    ax1.set_title('Key Performance Metrics vs Feature Count', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Number of Features', fontsize=14)
    ax1.set_ylabel('Score', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(feature_counts)

    # Right plot: Sensitivity and Specificity
    sn_values = [results[f'SHAP_{n}']['SN'] for n in feature_counts]
    sp_values = [results[f'SHAP_{n}']['SP'] for n in feature_counts]

    ax2.plot(feature_counts, sn_values, 'o-', linewidth=3, markersize=8,
             color='orange', label='Sensitivity (SN)')
    ax2.plot(feature_counts, sp_values, 'o-', linewidth=3, markersize=8,
             color='brown', label='Specificity (SP)')

    ax2.set_title('Sensitivity & Specificity vs Feature Count', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Number of Features', fontsize=14)
    ax2.set_ylabel('Score', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(feature_counts)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'combined_performance_overview.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'combined_performance_overview.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()


def create_roc_comparison(all_roc_data):
    """Create ROC curves comparison for different feature counts"""

    plt.figure(figsize=(12, 10))
    feature_counts = [200, 300, 500, 700, 900]
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for i, n_features in enumerate(feature_counts):
        method_name = f'SHAP_{n_features}'
        roc_data = all_roc_data[method_name]

        if roc_data and len(roc_data) > 0:
            if len(roc_data) >= 2:  # Binary classification
                fpr, tpr, auc_score = roc_data[1]  # Use positive class
            else:
                fpr, tpr, auc_score = roc_data[0]

            plt.plot(fpr, tpr, color=colors[i],
                     label=f'{n_features} features (AUC = {auc_score:.4f})',
                     linewidth=3)

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=16)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=16)
    plt.title('ROC Curves - Feature Count Comparison', fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'roc_curves_feature_count_comparison.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'roc_curves_feature_count_comparison.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()


def create_efficiency_analysis(results):
    """Create efficiency analysis plots"""

    feature_counts = [200, 300, 500, 700, 900]

    # Extract data
    accuracies = [results[f'SHAP_{n}']['ACC'] for n in feature_counts]
    training_times = [results[f'SHAP_{n}']['Training_Time'] for n in feature_counts]
    f1_scores = [results[f'SHAP_{n}']['F1'] for n in feature_counts]

    # Efficiency plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Accuracy per feature
    efficiency_acc = [acc / fc for acc, fc in zip(accuracies, feature_counts)]
    axes[0, 0].plot(feature_counts, efficiency_acc, 'o-', linewidth=3, markersize=8, color='blue')
    axes[0, 0].set_title('Accuracy Efficiency (Accuracy/Feature Count)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Features')
    axes[0, 0].set_ylabel('Accuracy per Feature')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(feature_counts)

    # Training time vs accuracy
    axes[0, 1].scatter(training_times, accuracies, s=100, c=feature_counts, cmap='viridis')
    for i, fc in enumerate(feature_counts):
        axes[0, 1].annotate(f'{fc}', (training_times[i], accuracies[i]),
                            xytext=(5, 5), textcoords='offset points')
    axes[0, 1].set_title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Training Time (seconds)')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)

    # Performance gain vs additional features
    performance_gains = [0] + [accuracies[i] - accuracies[i - 1] for i in range(1, len(accuracies))]
    feature_additions = [0] + [feature_counts[i] - feature_counts[i - 1] for i in range(1, len(feature_counts))]

    axes[1, 0].bar(range(len(feature_counts)), performance_gains, color='green', alpha=0.7)
    axes[1, 0].set_title('Performance Gain from Additional Features', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Feature Count Level')
    axes[1, 0].set_ylabel('Accuracy Gain')
    axes[1, 0].set_xticks(range(len(feature_counts)))
    axes[1, 0].set_xticklabels([f'{fc}' for fc in feature_counts])
    axes[1, 0].grid(True, alpha=0.3)

    # Time efficiency (performance per second)
    time_efficiency = [acc / time for acc, time in zip(accuracies, training_times)]
    axes[1, 1].plot(feature_counts, time_efficiency, 'o-', linewidth=3, markersize=8, color='red')
    axes[1, 1].set_title('Time Efficiency (Accuracy/Training Time)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Features')
    axes[1, 1].set_ylabel('Accuracy per Second')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(feature_counts)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'efficiency_analysis.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'efficiency_analysis.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()


def analyze_optimal_feature_count(results):
    """Analyze and recommend optimal feature count"""

    feature_counts = [200, 300, 500, 700, 900]

    print("\n" + "=" * 60)
    print("OPTIMAL FEATURE COUNT ANALYSIS")
    print("=" * 60)

    # Extract metrics
    metrics_data = {}
    for metric in ['ACC', 'AUC', 'F1', 'MCC', 'Training_Time']:
        metrics_data[metric] = [results[f'SHAP_{n}'][metric] for n in feature_counts]

    # Find optimal points
    max_acc_idx = np.argmax(metrics_data['ACC'])
    max_auc_idx = np.argmax(metrics_data['AUC'])
    max_f1_idx = np.argmax(metrics_data['F1'])
    max_mcc_idx = np.argmax(metrics_data['MCC'])
    min_time_idx = np.argmin(metrics_data['Training_Time'])

    print(f"Best Accuracy: {feature_counts[max_acc_idx]} features ({metrics_data['ACC'][max_acc_idx]:.4f})")
    print(f"Best AUC: {feature_counts[max_auc_idx]} features ({metrics_data['AUC'][max_auc_idx]:.4f})")
    print(f"Best F1-Score: {feature_counts[max_f1_idx]} features ({metrics_data['F1'][max_f1_idx]:.4f})")
    print(f"Best MCC: {feature_counts[max_mcc_idx]} features ({metrics_data['MCC'][max_mcc_idx]:.4f})")
    print(
        f"Fastest Training: {feature_counts[min_time_idx]} features ({metrics_data['Training_Time'][min_time_idx]:.2f}s)")

    # Calculate efficiency scores
    print(f"\n{'=' * 30}")
    print("EFFICIENCY ANALYSIS")
    print(f"{'=' * 30}")

    for i, fc in enumerate(feature_counts):
        acc_per_feature = metrics_data['ACC'][i] / fc
        acc_per_time = metrics_data['ACC'][i] / metrics_data['Training_Time'][i]
        print(f"{fc} features:")
        print(f"  Accuracy/Feature: {acc_per_feature:.6f}")
        print(f"  Accuracy/Time: {acc_per_time:.4f}")
        print(f"  Overall Score: {metrics_data['ACC'][i]:.4f}")
        print()

    # Performance plateau analysis
    print(f"{'=' * 30}")
    print("PERFORMANCE PLATEAU ANALYSIS")
    print(f"{'=' * 30}")

    acc_improvements = [metrics_data['ACC'][i] - metrics_data['ACC'][i - 1]
                        for i in range(1, len(metrics_data['ACC']))]

    for i, improvement in enumerate(acc_improvements):
        prev_fc = feature_counts[i]
        curr_fc = feature_counts[i + 1]
        additional_features = curr_fc - prev_fc
        print(f"{prev_fc} → {curr_fc} features: +{improvement:.4f} accuracy (+{additional_features} features)")
        print(f"  Improvement per added feature: {improvement / additional_features:.6f}")

    # Find diminishing returns point
    improvements_per_feature = [imp / (feature_counts[i + 1] - feature_counts[i])
                                for i, imp in enumerate(acc_improvements)]

    print(f"\nDiminishing returns analysis:")
    for i, ipf in enumerate(improvements_per_feature):
        print(f"  {feature_counts[i]} → {feature_counts[i + 1]}: {ipf:.6f} accuracy per feature")

    # Recommendation
    print(f"\n{'=' * 30}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 30}")

    # Find the best balance point
    normalized_acc = np.array(metrics_data['ACC']) / max(metrics_data['ACC'])
    normalized_time = 1 - (np.array(metrics_data['Training_Time']) / max(metrics_data['Training_Time']))
    normalized_efficiency = normalized_acc + normalized_time

    best_balance_idx = np.argmax(normalized_efficiency)

    print(f"🏆 OPTIMAL CHOICE (balanced performance/efficiency): {feature_counts[best_balance_idx]} features")
    print(f"   - Accuracy: {metrics_data['ACC'][best_balance_idx]:.4f}")
    print(f"   - Training Time: {metrics_data['Training_Time'][best_balance_idx]:.2f}s")
    print(f"   - Balance Score: {normalized_efficiency[best_balance_idx]:.4f}")

    print(f"\n⚡ FASTEST OPTION: {feature_counts[min_time_idx]} features")
    print(f"   - Accuracy: {metrics_data['ACC'][min_time_idx]:.4f}")
    print(f"   - Training Time: {metrics_data['Training_Time'][min_time_idx]:.2f}s")

    print(f"\n🎯 HIGHEST PERFORMANCE: {feature_counts[max_acc_idx]} features")
    print(f"   - Accuracy: {metrics_data['ACC'][max_acc_idx]:.4f}")
    print(f"   - Training Time: {metrics_data['Training_Time'][max_acc_idx]:.2f}s")


# Execute the comparison
if __name__ == "__main__":
    """
    Main function to execute the feature count comparison
    """
    print("Starting SHAP Feature Count Comparison...")
    print("=" * 80)
    print("SHAP FEATURE COUNT COMPARISON PIPELINE")
    print("=" * 80)

    try:
        # Run comparison
        print("🚀 Running feature count comparison...")
        results, selected_features, all_roc_data = run_feature_count_comparison()

        # Create results summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        results_df = pd.DataFrame(results).T
        print(results_df.round(4))

        # Save results
        print("\n💾 Saving results...")
        results_df.to_csv(os.path.join(BASE_DIR, 'feature_count_comparison_results.csv'))
        print(f"✅ Results saved to: {os.path.join(BASE_DIR, 'feature_count_comparison_results.csv')}")

        # Create visualizations
        print("\n📊 Creating comparison plots...")
        create_feature_count_comparison_plots(results)

        print("\n📈 Creating ROC curves...")
        create_roc_comparison(all_roc_data)

        print("\n⚡ Creating efficiency analysis...")
        create_efficiency_analysis(results)

        print("\n🔍 Analyzing optimal feature count...")
        analyze_optimal_feature_count(results)

        # Save selected features
        print("\n💾 Saving selected features...")
        for method, features in selected_features.items():
            feature_file = os.path.join(BASE_DIR, f'{method}_selected_features.npy')
            np.save(feature_file, features)
            print(f"✅ {method} features saved to: {feature_file}")

        # Feature count ranking
        print("\n" + "=" * 80)
        print("PERFORMANCE RANKING BY FEATURE COUNT")
        print("=" * 80)

        feature_counts = [200, 300, 500, 700, 900]
        accuracy_ranking = [(fc, results[f'SHAP_{fc}']['ACC']) for fc in feature_counts]
        accuracy_ranking.sort(key=lambda x: x[1], reverse=True)

        for i, (fc, acc) in enumerate(accuracy_ranking, 1):
            print(f"{i}. {fc} features: {acc:.4f} accuracy")

        # Key insights
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)

        best_fc, best_acc = accuracy_ranking[0]
        print(f"🏆 Best performing feature count: {best_fc} features ({best_acc:.4f} accuracy)")

        # Find most efficient
        efficiency_scores = [(fc, results[f'SHAP_{fc}']['ACC'] / results[f'SHAP_{fc}']['Training_Time'])
                             for fc in feature_counts]
        most_efficient = max(efficiency_scores, key=lambda x: x[1])
        print(f"⚡ Most efficient feature count: {most_efficient[0]} features ({most_efficient[1]:.4f} acc/sec)")

        # Find fastest
        time_ranking = [(fc, results[f'SHAP_{fc}']['Training_Time']) for fc in feature_counts]
        fastest = min(time_ranking, key=lambda x: x[1])
        print(f"🚀 Fastest training: {fastest[0]} features ({fastest[1]:.2f}s)")

        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        print("📊 Performance Range:")
        accuracies = [results[f'SHAP_{fc}']['ACC'] for fc in feature_counts]
        print(f"   Accuracy: {min(accuracies):.4f} - {max(accuracies):.4f}")

        times = [results[f'SHAP_{fc}']['Training_Time'] for fc in feature_counts]
        print(f"   Training Time: {min(times):.2f}s - {max(times):.2f}s")

        # Performance improvement analysis
        print(f"\n📈 Performance Improvement:")
        for i in range(1, len(feature_counts)):
            prev_acc = results[f'SHAP_{feature_counts[i - 1]}']['ACC']
            curr_acc = results[f'SHAP_{feature_counts[i]}']['ACC']
            improvement = curr_acc - prev_acc
            additional_features = feature_counts[i] - feature_counts[i - 1]
            print(
                f"   {feature_counts[i - 1]} → {feature_counts[i]} features: +{improvement:.4f} accuracy (+{additional_features} features)")

        # Final completion message
        print("\n" + "=" * 80)
        print("COMPLETION STATUS")
        print("=" * 80)

        print("✅ Feature count comparison completed successfully!")
        print(f"📁 All results saved to: {BASE_DIR}")
        print(f"📊 {len(feature_counts)} different feature counts tested")
        print(f"🎯 Feature counts tested: {', '.join(map(str, feature_counts))}")
        print(f"📋 Performance metrics calculated for each feature count")

        # File summary
        print("\n📁 Generated Files:")
        generated_files = [
            'feature_count_comparison_results.csv',
            'feature_count_performance_comparison.png',
            'combined_performance_overview.png',
            'roc_curves_feature_count_comparison.png',
            'training_time_vs_feature_count.png',
            'efficiency_analysis.png'
        ]

        for file in generated_files:
            file_path = os.path.join(BASE_DIR, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (not found)")

        # Feature importance plots
        print(f"\n📊 Feature Importance Plots:")
        for fc in feature_counts:
            importance_file = f'shap_importance_{fc}_features.png'
            file_path = os.path.join(BASE_DIR, importance_file)
            if os.path.exists(file_path):
                print(f"   ✅ {importance_file}")
            else:
                print(f"   ❌ {importance_file} (not found)")

        # Feature selection files
        print(f"\n📋 Feature Selection Files:")
        for fc in feature_counts:
            feature_file = f'SHAP_{fc}_selected_features.npy'
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

        print("\n🎉 Feature count comparison complete!")
        print("=" * 80)