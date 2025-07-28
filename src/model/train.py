import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    average_precision_score, precision_recall_curve
)
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

from model import build_lstm_model, focal_loss
from config import OUTPUT_DIR, SHAP_DIR, SHAP_SAMPLE_SIZE

# (source, horizon) pairs to compute SHAP for
SHAP_CONFIGS = {
    ('no_notes', 0), ('M1', 0),
    ('M2', 0), ('M2', 6),
    ('M3', 0), ('M3', 12)
}

def train_and_evaluate_lstm(data, model_name, source, horizon):
    variables, X_train, X_val, X_test, y_train, y_val, y_test = data

    # Compute class weights
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights_array))

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Build model
    model = build_lstm_model(input_shape=X_train.shape[1:])
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy', tf.keras.metrics.AUC()])

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        verbose=0,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )

    # Predict
    y_probs = model.predict(X_test).flatten()
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_preds = (y_probs >= best_threshold).astype(int)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
    specificity = tn / (tn + fp)

    metrics = {
        'accuracy': accuracy_score(y_test, y_preds),
        'precision': precision_score(y_test, y_preds),
        'specificity': specificity,
        'recall': recall_score(y_test, y_preds),
        'f1': f1_score(y_test, y_preds),
        'auc': roc_auc_score(y_test, y_probs),
        'prauc': average_precision_score(y_test, y_probs),
        'mcc': matthews_corrcoef(y_test, y_preds)
    }

    # Save loss history
    loss_df = pd.DataFrame(history.history)
    loss_df['epoch'] = range(1, len(loss_df) + 1)
    loss_df.to_csv(os.path.join(OUTPUT_DIR, f"M4_loss_history_{model_name}.csv"), index=False)

    # Attention outputs
    attention_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer("attn_weights").output,
            model.get_layer("max_pool").output
        ]
    )
    attn_weights, max_activations = attention_model.predict(X_test)
    pd.DataFrame(attn_weights).to_csv(os.path.join(OUTPUT_DIR, f"M4_attn_weights_{model_name}.csv"), index=False)
    pd.DataFrame(max_activations).to_csv(os.path.join(OUTPUT_DIR, f"M4_max_pool_features_{model_name}.csv"), index=False)
    print(f"✅ Saved attention and max pool outputs for {model_name}")

    # SHAP
    if (source, horizon) in SHAP_CONFIGS:
        try:
            n_samples = min(SHAP_SAMPLE_SIZE, X_test.shape[0])
            background = np.array(X_train[np.random.choice(X_train.shape[0], n_samples, replace=False)]).astype(np.float32)
            X_sample = np.array(X_test[:n_samples]).astype(np.float32)

            submodel = tf.keras.Model(inputs=model.input, outputs=model.get_layer("max_pool").output)
            background_flat = background.reshape(background.shape[0], -1)
            X_sample_flat = X_sample.reshape(X_sample.shape[0], -1)

            def wrapped_predict(X_flat):
                X_reshaped = X_flat.reshape((-1, background.shape[1], background.shape[2]))
                return submodel.predict(X_reshaped)

            shap_explainer = shap.KernelExplainer(wrapped_predict, background_flat)
            shap_values = shap_explainer.shap_values(X_sample_flat)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            pooled_features = submodel.predict(X_sample)
            pooled_feature_names = [f"feat_{i}" for i in range(pooled_features.shape[1])]

            shap.summary_plot(
                shap_values,
                features=pooled_features,
                feature_names=pooled_feature_names,
                show=False
            )
            plt.title(f"SHAP Summary (MaxPool): {model_name}")
            plt.savefig(os.path.join(SHAP_DIR, f"shap_{model_name}.png"))
            plt.close()
            print(f"✅ SHAP saved for {model_name}")
        except Exception as e:
            print(f"❌ SHAP failed for {model_name}: {e}")

    return metrics
