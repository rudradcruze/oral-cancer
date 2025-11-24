import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, matthews_corrcoef, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import catboost as cb
import time
import os
import warnings
import gc
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


warnings.filterwarnings('ignore')

# Set font family globally
plt.rcParams['font.family'] = 'Times New Roman'

# Base directory
BASE_DIR = "3_Model_Comparison/BEiT"
os.makedirs(BASE_DIR, exist_ok=True)

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

try:
    print("Loading and preparing data...")
    data = pd.read_csv('2_Feature_Extraction/BEiT/feature_extract_BEiT_features.csv')
    X = data.iloc[:, :1024].values  # Changed from 2048 to 1024 features
    y = data['label'].values  # Labels
    num_classes = len(np.unique(y))
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes")

    # Check class distribution
    print("Class distribution:")
    print(pd.Series(y).value_counts(normalize=True))
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



def create_data_loaders(X_train_subset, y_train_subset, X_val, y_val, batch_size=128):
    # Convert to PyTorch tensors and reshape
    X_train_3d = torch.FloatTensor(X_train_subset).unsqueeze(-1)
    X_val_3d = torch.FloatTensor(X_val).unsqueeze(-1)
    X_test_3d = torch.FloatTensor(X_test).unsqueeze(-1)
    y_train_t = torch.LongTensor(y_train_subset)
    y_val_t = torch.LongTensor(y_val)
    y_test_t = torch.LongTensor(y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train_3d, y_train_t)
    val_dataset = TensorDataset(X_val_3d, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, X_test_3d, y_test_t



# Model 1: Vision Transformer Head
class VisionTransformerHead(nn.Module):
    def __init__(self, num_classes, input_dim=1024):
        super(VisionTransformerHead, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = 128

        # Patch embedding
        self.patch_embed = nn.Linear(1, self.embed_dim)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim, self.embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        B, N, _ = x.shape  # [batch, 1024, 1]

        # Patch embedding
        x = self.patch_embed(x)  # [batch, 1024, 128]

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)  # [batch, 1024, 128]

        # Global average pooling
        x = x.mean(dim=1)  # [batch, 128]

        # Classification
        x = self.norm(x)
        x = self.head(x)  # [batch, num_classes]

        return x

class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes, input_dim=1024):
        super(MultiScaleCNN, self).__init__()

        # Enhanced multi-scale convolution branches with deeper architecture
        # Branch 1: Fine-grained features (small kernels)
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

        # Branch 4: Very large receptive field
        self.branch4 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=11, padding=5),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 160, kernel_size=11, padding=5),
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

        # Feature map size after convolutions: 1024 -> 512 -> 256
        self.feature_size = 256

        # Enhanced attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(embed_dim=160, num_heads=8, batch_first=True)

        # Fusion layers with residual connections
        self.fusion1 = nn.Sequential(
            nn.Conv1d(160 * 4 + 160, 512, kernel_size=1),  # 4 branches + global = 800 -> 512
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

        # Enhanced classification head with more capacity
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
        x = x.transpose(1, 2)  # [batch, 1, 1024]

        # Multi-scale feature extraction
        feat1 = self.branch1(x)  # [batch, 160, 256]
        feat2 = self.branch2(x)  # [batch, 160, 256]
        feat3 = self.branch3(x)  # [batch, 160, 256]
        feat4 = self.branch4(x)  # [batch, 160, 256]

        # Global context
        global_feat = self.global_branch(x)  # [batch, 160, 1]
        global_feat = global_feat.expand(-1, -1, self.feature_size)  # [batch, 160, 256]

        # Apply attention to each branch
        feat1_att = feat1.transpose(1, 2)  # [batch, 256, 160]
        feat1_att, _ = self.attention(feat1_att, feat1_att, feat1_att)
        feat1_att = feat1_att.transpose(1, 2)  # [batch, 160, 256]

        # Concatenate all features
        combined = torch.cat([feat1_att, feat2, feat3, feat4, global_feat], dim=1)  # [batch, 800, 256]

        # Multi-stage fusion
        fused1 = self.fusion1(combined)  # [batch, 512, 256]
        fused2 = self.fusion2(fused1)  # [batch, 256, 256]

        # Classification
        output = self.classifier(fused2)  # [batch, num_classes]

        return output


# Model 2: Attention-CNN (Enhanced version)
class AttentionCNN(nn.Module):
    def __init__(self, num_classes, input_dim=1024):
        super(AttentionCNN, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # Calculate the size after convolutions and pooling
        # 1024 -> 512 -> 256 -> 128 (after 3 pooling operations)
        self.fc_input_size = 128 * 256

        # Classification layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, 1, 1024]

        # CNN feature extraction
        x = F.relu(self.conv1(x))  # [batch, 64, 1024]
        x = self.pool(x)  # [batch, 64, 512]

        x = F.relu(self.conv2(x))  # [batch, 128, 512]
        x = self.pool(x)  # [batch, 128, 256]

        x = F.relu(self.conv3(x))  # [batch, 256, 256]
        x = self.pool(x)  # [batch, 256, 128]

        # Attention
        x = x.transpose(1, 2)  # [batch, 128, 256]
        x, _ = self.attention(x, x, x)  # [batch, 128, 256]

        # Flatten and classify
        x = x.reshape(x.size(0), -1)  # [batch, 128*256]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



# Focal Loss implementation for SVM
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# Model 4: Focal Loss SVM (Neural Network with SVM-like loss)
class FocalLossSVM(nn.Module):
    def __init__(self, num_classes, input_dim=1024):
        super(FocalLossSVM, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(-1)  # Remove the extra dimension [batch, 1024, 1] -> [batch, 1024]
        return self.layers(x)



def train_model(model, train_loader, val_loader, epochs=50, model_name="", use_focal_loss=False):
    try:
        model.to(device)

        if use_focal_loss:
            criterion = FocalLoss(alpha=1, gamma=2, num_classes=num_classes)
        else:
            class_weights = torch.tensor([1.0 / count for count in pd.Series(y_train).value_counts().sort_index()],
                                         dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        use_mixed_precision = device.type == "mps"
        scaler = GradScaler() if use_mixed_precision else None

        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                if use_mixed_precision:
                    with autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
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

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {history['loss'][-1]:.4f}, Val Loss: {val_loss:.4f}")

        return model, history
    except Exception as e:
        print(f"Error in training: {e}")
        raise



def evaluate_model(model, X_test_3d, y_test, name):
    """Evaluate a PyTorch model and return metrics"""
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        y_pred = model(X_test_3d.to(device)).cpu().numpy()
    test_time = time.time() - start_time
    y_pred_prob = torch.softmax(torch.tensor(y_pred), dim=1).numpy()
    y_pred_binary = np.argmax(y_pred_prob, axis=1)

    results = {
        'ACC': accuracy_score(y_test, y_pred_binary),
        'PRE': precision_score(y_test, y_pred_binary, average='macro'),
        'SP': recall_score(y_test, y_pred_binary, average='macro'),
        'SN': recall_score(y_test, y_pred_binary, average='macro'),
        'F1': f1_score(y_test, y_pred_binary, average='macro'),
        'MCC': matthews_corrcoef(y_test, y_pred_binary),
        'Testing Time': test_time
    }

    # AUC for multiclass (OvR)
    auc_scores = []
    roc_curves = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
        auc_scores.append(auc(fpr, tpr))
        roc_curves.append((fpr, tpr, auc_scores[-1]))

    results['AUC'] = np.mean(auc_scores)

    return results, roc_curves



def train_evaluate_catboost(X_train_subset, y_train_subset, X_val, y_val):
    """Train and evaluate CatBoost model"""
    print("\n------------------------------------------------")
    print("Training CatBoost Model")
    print("------------------------------------------------")

    start_time = time.time()

    # CatBoost parameters
    catboost_params = {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.1,
        'loss_function': 'MultiClass' if num_classes > 2 else 'Logloss',
        'eval_metric': 'MultiClass' if num_classes > 2 else 'Logloss',
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }

    if num_classes > 2:
        catboost_params['classes_count'] = num_classes

    catboost_model = cb.CatBoostClassifier(**catboost_params)

    # Train the model
    catboost_model.fit(
        X_train_subset, y_train_subset,
        eval_set=(X_val, y_val),
        verbose=False
    )

    train_time = time.time() - start_time

    # Predict
    start_time = time.time()
    y_pred_prob = catboost_model.predict_proba(X_test)
    test_time = time.time() - start_time

    y_pred_binary = np.argmax(y_pred_prob, axis=1)

    results = {
        'ACC': accuracy_score(y_test, y_pred_binary),
        'PRE': precision_score(y_test, y_pred_binary, average='macro'),
        'SP': recall_score(y_test, y_pred_binary, average='macro'),
        'SN': recall_score(y_test, y_pred_binary, average='macro'),
        'F1': f1_score(y_test, y_pred_binary, average='macro'),
        'MCC': matthews_corrcoef(y_test, y_pred_binary),
        'Training Time': train_time,
        'Testing Time': test_time
    }

    auc_scores = []
    roc_curves = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
        auc_scores.append(auc(fpr, tpr))
        roc_curves.append((fpr, tpr, auc_scores[-1]))

    results['AUC'] = np.mean(auc_scores)

    # Save CatBoost model
    model_path = os.path.join(BASE_DIR, 'catboost_model.cbm')
    catboost_model.save_model(model_path)
    print(f"CatBoost model saved to {model_path}")

    print("\nCatBoost Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return {'CatBoost': results}, {'CatBoost': roc_curves}



def train_evaluate_random_forest(X_train_subset, y_train_subset, X_val, y_val):
    """Train and evaluate Random Forest model"""
    print("\n------------------------------------------------")
    print("Training Random Forest Model")
    print("------------------------------------------------")

    start_time = time.time()

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    rf_model.fit(X_train_subset, y_train_subset)
    train_time = time.time() - start_time

    # Predict
    start_time = time.time()
    y_pred_prob = rf_model.predict_proba(X_test)
    test_time = time.time() - start_time

    y_pred_binary = np.argmax(y_pred_prob, axis=1)

    results = {
        'ACC': accuracy_score(y_test, y_pred_binary),
        'PRE': precision_score(y_test, y_pred_binary, average='macro'),
        'SP': recall_score(y_test, y_pred_binary, average='macro'),
        'SN': recall_score(y_test, y_pred_binary, average='macro'),
        'F1': f1_score(y_test, y_pred_binary, average='macro'),
        'MCC': matthews_corrcoef(y_test, y_pred_binary),
        'Training Time': train_time,
        'Testing Time': test_time
    }

    auc_scores = []
    roc_curves = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
        auc_scores.append(auc(fpr, tpr))
        roc_curves.append((fpr, tpr, auc_scores[-1]))

    results['AUC'] = np.mean(auc_scores)

    print("\nRandom Forest Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return {'Random Forest': results}, {'Random Forest': roc_curves}



def plot_loss_curves(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], color='blue', linewidth=2, label='Training Loss')
    plt.plot(history['val_loss'], color='red', linewidth=2, label='Validation Loss')

    plt.title(f'{model_name} Loss Curves', fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    safe_name = model_name.replace(' ', '_').lower()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f'{safe_name}_loss_curves.png'), dpi=1000)
    plt.savefig(os.path.join(BASE_DIR, f'{safe_name}_loss_curves.pdf'))
    plt.close()



def run_pytorch_model(model_name, model_class, use_focal_loss=False):
    print(f"\nTraining {model_name} Model")
    try:
        X_train_subset, X_val, y_train_subset, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                                        random_state=101)
        train_loader, val_loader, X_test_3d, y_test_t = create_data_loaders(X_train_subset, y_train_subset,
                                                                            X_val, y_val, batch_size=64)

        model = model_class(num_classes)
        start_time = time.time()
        model, history = train_model(model, train_loader, val_loader, epochs=50,
                                     model_name=model_name, use_focal_loss=use_focal_loss)
        train_time = time.time() - start_time

        plot_loss_curves(history, model_name)

        results, roc_curves = evaluate_model(model, X_test_3d, y_test, model_name)
        results['Training Time'] = train_time

        # Save model and history
        model_path = os.path.join(BASE_DIR, f"{model_name.replace(' ', '_').lower()}.pt")
        torch.save(model.state_dict(), model_path)
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(BASE_DIR, f"{model_name.replace(' ', '_').lower()}_history.csv"))

        del model, train_loader, val_loader, X_test_3d, y_test_t
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        print(f"\n{model_name} Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        return {model_name: results}, {model_name: roc_curves}
    except Exception as e:
        print(f"Error in running {model_name}: {e}")
        return {model_name: {'Error': str(e)}}, {model_name: []}



def plot_combined_loss_curves():
    """Create a combined plot showing the validation loss curves for all models"""
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'green', 'red', 'purple', 'orange']
    models = ['Vision Transformer Head', 'Attention-CNN', 'Focal Loss SVM']

    for i, model_name in enumerate(models):
        try:
            safe_name = model_name.replace(' ', '_').lower()
            history_path = os.path.join(BASE_DIR, f"{safe_name}_history.csv")

            if os.path.exists(history_path):
                history_df = pd.read_csv(history_path)
                plt.plot(history_df['val_loss'],
                         color=colors[i % len(colors)],
                         linewidth=2,
                         label=f'{model_name} Val Loss')
        except Exception as e:
            print(f"Error loading history for {model_name}: {e}")

    plt.title('Validation Loss Comparison Across Models', fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Validation Loss', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'combined_val_loss_comparison.png'), dpi=1000)
    plt.savefig(os.path.join(BASE_DIR, 'combined_val_loss_comparison.pdf'), dpi=1000)
    plt.close()



# Main execution
all_results = {}
all_roc_data = {}

# Prepare data split for traditional ML models
X_train_subset, X_val, y_train_subset, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=101)

# Run PyTorch models
for model_name, model_class, use_focal in [
    ('Vision Transformer', VisionTransformerHead, False),
    ('Attention-CNN', AttentionCNN, True),
    ('Focal Loss SVM', FocalLossSVM, True),
]:
    try:
        results, roc_data = run_pytorch_model(model_name, model_class, use_focal_loss=use_focal)
        all_results.update(results)
        all_roc_data.update(roc_data)
    except Exception as e:
        print(f"Error running {model_name}: {e}")
        all_results[model_name] = {'Error': str(e)}

# Run CatBoost
try:
    catboost_results, catboost_roc = train_evaluate_catboost(X_train_subset, y_train_subset, X_val, y_val)
    all_results.update(catboost_results)
    all_roc_data.update(catboost_roc)
except Exception as e:
    print(f"Error running CatBoost: {e}")
    all_results['CatBoost'] = {'Error': str(e)}

# Run Random Forest
try:
    rf_results, rf_roc = train_evaluate_random_forest(X_train_subset, y_train_subset, X_val, y_val)
    all_results.update(rf_results)
    all_roc_data.update(rf_roc)
except Exception as e:
    print(f"Error running Random Forest: {e}")
    all_results['Random Forest'] = {'Error': str(e)}

# Create ROC curve plots
plt.figure(figsize=(10, 8))
plt.gca().set_facecolor('white')
colors = ['blue', 'green', 'red', 'purple', 'orange']

for i, (model_name, roc_data) in enumerate(all_roc_data.items()):
    if roc_data:
        for j, (fpr, tpr, auc_value) in enumerate(roc_data):
            if j == 0:  # Only show first class for multiclass
                plt.plot(fpr, tpr, color=colors[i % len(colors)],
                         label=f'{model_name} (AUC = {auc_value:.4f})', linewidth=3)

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

ax.patch.set_linewidth(2)
ax.patch.set_edgecolor('black')

plt.xlabel('False Positive Rate', fontsize=24)
plt.ylabel('True Positive Rate', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('ROC Curves for All Models', fontsize=28, pad=20)
plt.legend(loc='lower right', fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'all_models_roc.png'), dpi=1000)
plt.savefig(os.path.join(BASE_DIR, 'all_models_roc.pdf'), dpi=1000)
plt.show()
plt.close()

# Summary
print("\nSummary of All Models:")
summary_df = pd.DataFrame(all_results).T
column_order = ['ACC', 'AUC', 'PRE', 'SP', 'SN', 'F1', 'MCC', 'Training Time', 'Testing Time']
summary_df = summary_df.reindex(columns=column_order)
summary_df.to_csv(os.path.join(BASE_DIR, 'all_models_summary.csv'))
print(summary_df.round(4))

print("\nGenerating combined validation loss curves...")
plot_combined_loss_curves()
print("Done! All loss curves have been saved to the output directory.")

# Final memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("All done!")