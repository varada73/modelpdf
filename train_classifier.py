import pandas as pd
import lightgbm as lgb
import joblib
import json
import numpy as np

# --- 1. Configuration ---
FEATURES_FILE = "features.parquet"
MODEL_OUTPUT_FILE = "lgbm_classifier.joblib"
MODEL_FEATURES_FILE = "lgbm_features.json"
LABEL_MAP_FILE = "label_map.json"

# --- 2. Load Data and Prepare for Training ---
print(f"Loading data from {FEATURES_FILE}...")
df = pd.read_parquet(FEATURES_FILE)

# --- Oversampling Logic ---
print("Balancing the dataset via oversampling...")
max_size = df['label'].value_counts().max()
lst = [df]
for class_index, group in df.groupby('label'):
    if len(group) < max_size:
        lst.append(group.sample(max_size - len(group), replace=True, random_state=42))
df_balanced = pd.concat(lst)

# --- Create Label Mappings ---
unique_labels = sorted(df_balanced['label'].unique().tolist())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}

# --- Prepare data for training ---
text_embeddings = np.array(df_balanced['text_embedding'].tolist())
visual_features = df_balanced.drop(columns=['text', 'label', 'text_embedding']).values

X = np.concatenate([text_embeddings, visual_features], axis=1).astype(np.float32)
y = df_balanced['label'].map(label2id)

feature_names = [f'embed_{i}' for i in range(text_embeddings.shape[1])] + df_balanced.drop(columns=['text', 'label', 'text_embedding']).columns.tolist()

# --- 3. Train the LightGBM Classifier ---
print("Training LightGBM classifier...")
lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=42)
lgbm.fit(X, y)

# --- 4. Save the Model and Supporting Files ---
joblib.dump(lgbm, MODEL_OUTPUT_FILE)
with open(MODEL_FEATURES_FILE, 'w') as f:
    json.dump(feature_names, f)
with open(LABEL_MAP_FILE, 'w') as f:
    json.dump({'id2label': id2label}, f)
    
print(f"Classifier saved to '{MODEL_OUTPUT_FILE}'")
print(f"Feature list and label map saved.")