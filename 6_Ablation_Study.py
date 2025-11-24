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
BASE_DIR = "6_Ablation_Study/SwinTransformer_Ablation"
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


# Apply SHAP feature selection (use 500 features as baseline)
def get_selected_features():
    """Get SHAP-selected features for consistent comparison"""
    print("Applying SHAP feature selection for 500 features...")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf_model)
    max_samples = min(100, X_train.shape[0])
    sample_indices = np.random.choice(X_train.shape[0], max_samples, replace=False)
    X_shap_sample = X_train[sample_indices]

    shap_values = explainer.shap_values(X_shap_sample)

    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_vals_for_analysis = shap_values[1]
        else:
            shap_vals_for_analysis = np.mean(shap_values, axis=0)
    else:
        shap_vals_for_analysis = shap_values

    if len(shap_vals_for_analysis.shape) > 2:
        if shap_vals_for_analysis.shape[-1] == 2:
            shap_vals_for_analysis = shap_vals_for_analysis[:, :, 1]
        else:
            shap_vals_for_analysis = shap_vals_for_analysis.reshape(shap_vals_for_analysis.shape[0], -1)

    if len(shap_vals_for_analysis.shape) == 2:
        mean_shap_values = np.abs(shap_vals_for_analysis).mean(axis=0)
    else:
        mean_shap_values = rf_model.feature_importances_

    if len(mean_shap_values) != X_train.shape[1]:
        mean_shap_values = rf_model.feature_importances_

    selected_indices = np.argsort(mean_shap_values)[-500:]
    print(f"Selected {len(selected_indices)} features using SHAP")
    return selected_indices


# Get selected features
selected_feature_indices = get_selected_features()
X_train_selected = X_train[:, selected_feature_indices]
X_val_selected = X_val[:, selected_feature_indices]
X_test_selected = X_test[:, selected_feature_indices]

print(f"Using {X_train_selected.shape[1]} selected features for ablation study")


# ============================================================================
# MODEL VARIATIONS FOR ABLATION STUDY
# ============================================================================

# 1. Full Model (Complete MultiScaleCNN)
class FullMultiScaleCNN(nn.Module):
    """Complete model with all components"""

    def __init__(self, num_classes, input_dim=500):
        super(FullMultiScaleCNN, self).__init__()
        self.input_dim = input_dim
        self.feature_size = input_dim // 4

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


# 2. No Attention Model
class NoAttentionCNN(nn.Module):
    """Model without attention mechanism"""

    def __init__(self, num_classes, input_dim=500):
        super(NoAttentionCNN, self).__init__()
        self.input_dim = input_dim
        self.feature_size = input_dim // 4

        # Same branches as full model
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

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(1, 160, kernel_size=1),
            nn.BatchNorm1d(160),
            nn.ReLU()
        )

        # NO ATTENTION - direct fusion
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
        x = x.transpose(1, 2)

        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)

        global_feat = self.global_branch(x)
        global_feat = global_feat.expand(-1, -1, self.feature_size)

        # Direct concatenation without attention
        combined = torch.cat([feat1, feat2, feat3, global_feat], dim=1)

        fused1 = self.fusion1(combined)
        fused2 = self.fusion2(fused1)
        output = self.classifier(fused2)

        return output


# 3. Single Scale Model (only one branch)
class SingleScaleCNN(nn.Module):
    """Model with only one scale (kernel size 3)"""

    def __init__(self, num_classes, input_dim=500):
        super(SingleScaleCNN, self).__init__()
        self.input_dim = input_dim
        self.feature_size = input_dim // 4

        # Only one branch (kernel size 3)
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

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(1, 160, kernel_size=1),
            nn.BatchNorm1d(160),
            nn.ReLU()
        )

        self.attention = nn.MultiheadAttention(embed_dim=160, num_heads=8, batch_first=True)

        # Adjusted for single branch
        self.fusion1 = nn.Sequential(
            nn.Conv1d(160 + 160, 512, kernel_size=1),  # Only one branch + global
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
        x = x.transpose(1, 2)

        feat1 = self.branch1(x)
        global_feat = self.global_branch(x)
        global_feat = global_feat.expand(-1, -1, self.feature_size)

        # Apply attention
        feat1_att = feat1.transpose(1, 2)
        feat1_att, _ = self.attention(feat1_att, feat1_att, feat1_att)
        feat1_att = feat1_att.transpose(1, 2)

        # Combine only single branch with global
        combined = torch.cat([feat1_att, global_feat], dim=1)

        fused1 = self.fusion1(combined)
        fused2 = self.fusion2(fused1)
        output = self.classifier(fused2)

        return output


# 4. No Global Context Model
class NoGlobalContextCNN(nn.Module):
    """Model without global context branch"""

    def __init__(self, num_classes, input_dim=500):
        super(NoGlobalContextCNN, self).__init__()
        self.input_dim = input_dim
        self.feature_size = input_dim // 4

        # Three branches but no global context
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

        # NO GLOBAL CONTEXT BRANCH

        self.attention = nn.MultiheadAttention(embed_dim=160, num_heads=8, batch_first=True)

        # Adjusted for no global context
        self.fusion1 = nn.Sequential(
            nn.Conv1d(160 * 3, 512, kernel_size=1),  # Only three branches
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
        x = x.transpose(1, 2)

        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)

        # Apply attention to first branch
        feat1_att = feat1.transpose(1, 2)
        feat1_att, _ = self.attention(feat1_att, feat1_att, feat1_att)
        feat1_att = feat1_att.transpose(1, 2)

        # Combine only the three branches (no global context)
        combined = torch.cat([feat1_att, feat2, feat3], dim=1)

        fused1 = self.fusion1(combined)
        fused2 = self.fusion2(fused1)
        output = self.classifier(fused2)

        return output


# 5. Simple CNN (Baseline)
class SimpleCNN(nn.Module):
    """Simple baseline CNN for comparison"""

    def __init__(self, num_classes, input_dim=500):
        super(SimpleCNN, self).__init__()
        self.input_dim = input_dim

        # Simple sequential CNN
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def create_data_loaders(X_train_subset, y_train_subset, X_val, y_val, batch_size=32):
    X_train_3d = torch.FloatTensor(X_train_subset).unsqueeze(-1)
    X_val_3d = torch.FloatTensor(X_val).unsqueeze(-1)
    X_test_3d = torch.FloatTensor(X_test_selected).unsqueeze(-1)
    y_train_t = torch.LongTensor(y_train_subset)
    y_val_t = torch.LongTensor(y_val)
    y_test_t = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_3d, y_train_t)
    val_dataset = TensorDataset(X_val_3d, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, X_test_3d, y_test_t


def train_model(model, train_loader, val_loader, epochs=25, model_name="Model"):
    model.to(device)
    print(f"Training {model_name}...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    history = {'loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_acc = 100 * train_correct / train_total
        history['loss'].append(train_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - Train Loss: {history['loss'][-1]:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print(f"  {model_name} - Best Validation Accuracy: {best_val_acc:.2f}%")
    return model, history


def evaluate_model(model, X_test_selected, y_test, model_name="Model"):
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

    print(f"  {model_name} Test Results:")
    print(f"    Accuracy: {results['ACC']:.4f}")
    print(f"    AUC: {results['AUC']:.4f}")
    print(f"    F1-Score: {results['F1']:.4f}")
    print(f"    MCC: {results['MCC']:.4f}")

    return results, roc_curves


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# MAIN ABLATION STUDY FUNCTION
# ============================================================================

def run_ablation_study():
    print("=" * 80)
    print("MULTISCALE CNN ABLATION STUDY")
    print("=" * 80)

    # Define model variations for ablation study
    model_variants = {
        'Full_Model': {
            'class': FullMultiScaleCNN,
            'description': 'Complete MultiScaleCNN with all components'
        },
        'No_Attention': {
            'class': NoAttentionCNN,
            'description': 'MultiScaleCNN without attention mechanism'
        },
        'Single_Scale': {
            'class': SingleScaleCNN,
            'description': 'Single scale CNN (kernel size 3 only)'
        },
        'No_Global_Context': {
            'class': NoGlobalContextCNN,
            'description': 'MultiScaleCNN without global context branch'
        },
        'Simple_CNN': {
            'class': SimpleCNN,
            'description': 'Simple baseline CNN'
        }
    }

    # Store results
    all_results = {}
    all_roc_data = {}
    all_histories = {}
    parameter_counts = {}
    training_times = {}

    print(f"Using {X_train_selected.shape[1]} features selected by SHAP")
    print(f"Training dataset: {X_train_selected.shape[0]} samples")
    print(f"Validation dataset: {X_val_selected.shape[0]} samples")
    print(f"Test dataset: {X_test_selected.shape[0]} samples")

    # Run ablation study for each model variant
    for variant_name, variant_info in model_variants.items():
        print(f"\n{'=' * 60}")
        print(f"TESTING: {variant_name}")
        print(f"Description: {variant_info['description']}")
        print(f"{'=' * 60}")

        try:
            # Initialize model
            model = variant_info['class'](num_classes, input_dim=X_train_selected.shape[1])

            # Count parameters
            param_count = count_parameters(model)
            parameter_counts[variant_name] = param_count
            print(f"Model parameters: {param_count:,}")

            # Create data loaders
            train_loader, val_loader, _, _ = create_data_loaders(
                X_train_selected, y_train, X_val_selected, y_val
            )

            # Train model
            start_time = time.time()
            model, history = train_model(model, train_loader, val_loader,
                                         epochs=25, model_name=variant_name)
            training_time = time.time() - start_time
            training_times[variant_name] = training_time

            # Evaluate model
            results, roc_curves = evaluate_model(model, X_test_selected, y_test, variant_name)
            results['Training_Time'] = training_time
            results['Parameters'] = param_count

            # Store results
            all_results[variant_name] = results
            all_roc_data[variant_name] = roc_curves
            all_histories[variant_name] = history

            print(f"Training completed in {training_time:.2f} seconds")

            # Clear memory
            del model, train_loader, val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            print(f"Error training {variant_name}: {e}")
            continue

    return all_results, all_roc_data, all_histories, parameter_counts, training_times


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_ablation_comparison_plots(results, parameter_counts):
    """Create comprehensive comparison plots for ablation study"""

    # Extract data
    models = list(results.keys())
    metrics = ['ACC', 'AUC', 'PRE', 'SP', 'SN', 'F1', 'MCC']

    # Main performance comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]

        bars = axes[i].bar(models, values, color=colors)
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
    plt.savefig(os.path.join(BASE_DIR, 'ablation_performance_comparison.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'ablation_performance_comparison.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()

    # Model complexity vs performance
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Parameters vs Accuracy
    params = [parameter_counts[model] for model in models]
    accuracies = [results[model]['ACC'] for model in models]

    scatter = ax1.scatter(params, accuracies, s=150, c=range(len(models)), cmap='viridis', alpha=0.7)
    for i, model in enumerate(models):
        ax1.annotate(model.replace('_', '\n'), (params[i], accuracies[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax1.set_xlabel('Number of Parameters', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Complexity vs Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Training time vs Accuracy
    times = [results[model]['Training_Time'] for model in models]

    scatter = ax2.scatter(times, accuracies, s=150, c=range(len(models)), cmap='plasma', alpha=0.7)
    for i, model in enumerate(models):
        ax2.annotate(model.replace('_', '\n'), (times[i], accuracies[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Training Time (seconds)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Time vs Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Parameter efficiency (Accuracy per parameter)
    param_efficiency = [acc / param * 1000000 for acc, param in zip(accuracies, params)]  # per million params

    bars = ax3.bar(models, param_efficiency, color=colors)
    ax3.set_title('Parameter Efficiency (Accuracy per Million Parameters)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Accuracy per Million Parameters', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    for bar, value in zip(bars, param_efficiency):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(param_efficiency) * 0.01,
                 f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    # Time efficiency (Accuracy per second)
    time_efficiency = [acc / time for acc, time in zip(accuracies, times)]

    bars = ax4.bar(models, time_efficiency, color=colors)
    ax4.set_title('Time Efficiency (Accuracy per Second)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy per Second', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    for bar, value in zip(bars, time_efficiency):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(time_efficiency) * 0.01,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'ablation_efficiency_analysis.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'ablation_efficiency_analysis.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()


def create_ablation_roc_curves(all_roc_data):
    """Create ROC curves for all model variants"""

    plt.figure(figsize=(12, 10))
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, (model_name, roc_data) in enumerate(all_roc_data.items()):
        if roc_data and len(roc_data) > 0:
            if len(roc_data) >= 2:  # Binary classification
                fpr, tpr, auc_score = roc_data[1]  # Use positive class
            else:
                fpr, tpr, auc_score = roc_data[0]

            plt.plot(fpr, tpr, color=colors[i % len(colors)],
                     label=f'{model_name.replace("_", " ")} (AUC = {auc_score:.4f})',
                     linewidth=3)

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=16)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=16)
    plt.title('ROC Curves - Ablation Study Comparison', fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'ablation_roc_curves.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'ablation_roc_curves.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()


def create_component_contribution_analysis(results):
    """Analyze the contribution of each component"""

    # Define component contributions
    full_model_acc = results['Full_Model']['ACC']

    component_analysis = {
        'Attention Mechanism': full_model_acc - results['No_Attention']['ACC'],
        'Multi-Scale Branches': full_model_acc - results['Single_Scale']['ACC'],
        'Global Context': full_model_acc - results['No_Global_Context']['ACC'],
        'Advanced Architecture': full_model_acc - results['Simple_CNN']['ACC']
    }

    # Create contribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Component contribution bars
    components = list(component_analysis.keys())
    contributions = list(component_analysis.values())
    colors = ['red' if x < 0 else 'green' for x in contributions]

    bars = ax1.bar(components, contributions, color=colors, alpha=0.7)
    ax1.set_title('Component Contribution to Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy Contribution', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add value labels
    for bar, value in zip(bars, contributions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + (0.001 if height >= 0 else -0.002),
                 f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=11)

    # Model performance ranking
    model_performance = [(name.replace('_', ' '), results[name]['ACC']) for name in results.keys()]
    model_performance.sort(key=lambda x: x[1], reverse=True)

    models, accuracies = zip(*model_performance)
    bars = ax2.bar(models, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    ax2.set_title('Model Performance Ranking', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'component_contribution_analysis.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'component_contribution_analysis.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()

    return component_analysis


def create_training_curves(all_histories):
    """Create training and validation curves for all models"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    # Plot training and validation loss
    for i, (model_name, history) in enumerate(all_histories.items()):
        row = i // 3
        col = i % 3

        if row < 2 and col < 3:
            epochs = range(1, len(history['loss']) + 1)

            axes[row, col].plot(epochs, history['loss'], color=colors[i % len(colors)],
                                linewidth=2, label='Training Loss')
            axes[row, col].plot(epochs, history['val_loss'], color=colors[i % len(colors)],
                                linewidth=2, linestyle='--', label='Validation Loss')

            if 'val_acc' in history:
                ax2 = axes[row, col].twinx()
                ax2.plot(epochs, history['val_acc'], color='black',
                         linewidth=2, linestyle=':', label='Val Accuracy')
                ax2.set_ylabel('Validation Accuracy (%)', fontsize=10)
                ax2.legend(loc='upper right')

            axes[row, col].set_title(f'{model_name.replace("_", " ")}', fontsize=12, fontweight='bold')
            axes[row, col].set_xlabel('Epoch', fontsize=10)
            axes[row, col].set_ylabel('Loss', fontsize=10)
            axes[row, col].legend(loc='upper left')
            axes[row, col].grid(True, alpha=0.3)

    # Hide unused subplot if any
    if len(all_histories) < 6:
        for i in range(len(all_histories), 6):
            row = i // 3
            col = i % 3
            if row < 2 and col < 3:
                axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_curves_comparison.png'), dpi=1000, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'training_curves_comparison.pdf'), dpi=1000, bbox_inches='tight')
    plt.show()


def analyze_ablation_insights(results, component_analysis, parameter_counts):
    """Generate detailed insights from ablation study"""

    print("\n" + "=" * 80)
    print("ABLATION STUDY INSIGHTS")
    print("=" * 80)

    # Model rankings
    model_ranking = sorted(results.items(), key=lambda x: x[1]['ACC'], reverse=True)

    print("\n📊 MODEL PERFORMANCE RANKING:")
    print("-" * 50)
    for i, (model, result) in enumerate(model_ranking, 1):
        print(f"{i}. {model.replace('_', ' ')}: {result['ACC']:.4f} accuracy")
        print(f"   - Parameters: {parameter_counts[model]:,}")
        print(f"   - Training Time: {result['Training_Time']:.2f}s")
        print(f"   - F1-Score: {result['F1']:.4f}")
        print(f"   - AUC: {result['AUC']:.4f}")
        print()

    # Component contributions
    print("\n🔍 COMPONENT CONTRIBUTION ANALYSIS:")
    print("-" * 50)
    for component, contribution in component_analysis.items():
        impact = "Positive" if contribution > 0 else "Negative" if contribution < 0 else "Neutral"
        print(f"{component}: {contribution:+.4f} accuracy ({impact} impact)")

    # Key insights
    print("\n💡 KEY INSIGHTS:")
    print("-" * 50)

    best_model = model_ranking[0][0]
    best_acc = model_ranking[0][1]['ACC']
    print(f"🏆 Best Model: {best_model.replace('_', ' ')} ({best_acc:.4f} accuracy)")

    # Find most efficient model
    efficiency_scores = [(name, results[name]['ACC'] / results[name]['Training_Time'])
                         for name in results.keys()]
    most_efficient = max(efficiency_scores, key=lambda x: x[1])
    print(f"⚡ Most Efficient: {most_efficient[0].replace('_', ' ')} ({most_efficient[1]:.4f} acc/sec)")

    # Find smallest model
    smallest_model = min(parameter_counts.items(), key=lambda x: x[1])
    print(f"📦 Smallest Model: {smallest_model[0].replace('_', ' ')} ({smallest_model[1]:,} parameters)")

    # Component importance ranking
    component_importance = sorted(component_analysis.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n🎯 COMPONENT IMPORTANCE RANKING:")
    for i, (component, contribution) in enumerate(component_importance, 1):
        print(f"{i}. {component}: {abs(contribution):.4f} absolute impact")

    # Performance vs complexity trade-offs
    print(f"\n⚖️ PERFORMANCE VS COMPLEXITY TRADE-OFFS:")
    print("-" * 50)

    for model_name, result in results.items():
        param_count = parameter_counts[model_name]
        acc_per_param = result['ACC'] / param_count * 1000000  # per million params
        print(f"{model_name.replace('_', ' ')}:")
        print(f"  Accuracy/Parameter efficiency: {acc_per_param:.2f} per million params")
        print(f"  Time efficiency: {result['ACC'] / result['Training_Time']:.4f} acc/sec")

    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("-" * 50)

    # Best overall
    print(f"• For highest performance: Use {best_model.replace('_', ' ')}")

    # Best efficient
    print(f"• For best efficiency: Use {most_efficient[0].replace('_', ' ')}")

    # Resource constrained
    print(f"• For resource constraints: Use {smallest_model[0].replace('_', ' ')}")

    # Component recommendations
    most_important_component = component_importance[0][0]
    print(f"• Most critical component: {most_important_component}")

    if component_analysis['Attention Mechanism'] > 0.01:
        print("• Attention mechanism provides significant benefit - keep it")
    elif component_analysis['Attention Mechanism'] < -0.01:
        print("• Attention mechanism hurts performance - consider removing")
    else:
        print("• Attention mechanism has minimal impact")

    if component_analysis['Multi-Scale Branches'] > 0.01:
        print("• Multi-scale architecture is beneficial - keep multiple branches")
    elif component_analysis['Multi-Scale Branches'] < -0.01:
        print("• Multi-scale architecture hurts performance - use single scale")
    else:
        print("• Multi-scale architecture has minimal impact")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main function to execute the complete ablation study
    """
    print("Starting MultiScaleCNN Ablation Study...")
    print("=" * 80)
    print("ABLATION STUDY PIPELINE INITIATED")
    print("=" * 80)

    try:
        # Run ablation study
        print("🚀 Running ablation study...")
        results, all_roc_data, all_histories, parameter_counts, training_times = run_ablation_study()

        # Create results summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        results_df = pd.DataFrame(results).T
        print(results_df.round(4))

        # Save results
        print("\n💾 Saving results...")
        results_df.to_csv(os.path.join(BASE_DIR, 'ablation_study_results.csv'))
        print(f"✅ Results saved to: {os.path.join(BASE_DIR, 'ablation_study_results.csv')}")

        # Create visualizations
        print("\n📊 Creating comparison plots...")
        create_ablation_comparison_plots(results, parameter_counts)

        print("\n📈 Creating ROC curves...")
        create_ablation_roc_curves(all_roc_data)

        print("\n🔍 Analyzing component contributions...")
        component_analysis = create_component_contribution_analysis(results)

        print("\n📋 Creating training curves...")
        create_training_curves(all_histories)

        print("\n🧠 Generating insights...")
        analyze_ablation_insights(results, component_analysis, parameter_counts)

        # Save parameter counts
        print("\n💾 Saving parameter counts...")
        param_df = pd.DataFrame.from_dict(parameter_counts, orient='index', columns=['Parameters'])
        param_df.to_csv(os.path.join(BASE_DIR, 'model_parameter_counts.csv'))

        # Final completion message
        print("\n" + "=" * 80)
        print("COMPLETION STATUS")
        print("=" * 80)

        print("✅ Ablation study completed successfully!")
        print(f"📁 All results saved to: {BASE_DIR}")
        print(f"🔬 {len(results)} model variants tested")
        print(f"📊 Component contribution analysis completed")
        print(f"📈 Training curves generated for all models")

        # File summary
        print("\n📁 Generated Files:")
        generated_files = [
            'ablation_study_results.csv',
            'ablation_performance_comparison.png',
            'ablation_efficiency_analysis.png',
            'ablation_roc_curves.png',
            'component_contribution_analysis.png',
            'training_curves_comparison.png',
            'model_parameter_counts.csv'
        ]

        for file in generated_files:
            file_path = os.path.join(BASE_DIR, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (not found)")

        print(f"\n🎯 Study tested the following components:")
        print(f"   • Full MultiScaleCNN (complete architecture)")
        print(f"   • No Attention variant (removed attention mechanism)")
        print(f"   • Single Scale variant (single kernel size)")
        print(f"   • No Global Context variant (removed global branch)")
        print(f"   • Simple CNN baseline (basic architecture)")

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

        print("\n🎉 Ablation study complete!")
        print("=" * 80)