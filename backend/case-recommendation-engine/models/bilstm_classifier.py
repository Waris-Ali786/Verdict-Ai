import numpy as np
import os
from typing import List, Dict, Tuple, Optional

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dense, Dropout, BatchNormalization, SpatialDropout1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from data.preprocessor import TextPreprocessor
from data.cases_dataset import LegalDataset


# ──────────────────────────────────────────────────
# Updated constants for real dataset
# ──────────────────────────────────────────────────

CASE_TYPES = [
    "criminal", "family", "civil",
    "corporate", "constitutional", "service", "tax",
]

MAX_VOCAB_SIZE = 30_000   # v1: 10,000  — real judgments have ~80k unique terms
MAX_SEQ_LEN   = 512       # v1: 200     — Supreme Court judgments are long
EMBEDDING_DIM = 128       # v1: 64      — larger embedding for richer vocabulary
LSTM_UNITS    = 128       # v1: 64      — more capacity for complex legal text
DROPOUT_RATE  = 0.4       # v1: 0.3    — slightly higher for larger model
EPOCHS        = 20        # v1: 15
BATCH_SIZE    = 32        # v1: 8       — can use larger batch with 1414 samples


class BiLSTMClassifier:
    """
    Bidirectional LSTM for legal case type classification.
    Updated for 1,414 real Supreme Court of Pakistan judgments.

    Now classifies 7 case types:
      criminal | family | civil | corporate | constitutional | service | tax

    Samsung curriculum:
      - TensorFlow / Keras model building
      - Embedding layers, BiLSTM, Dropout, BatchNorm
      - Training with callbacks, evaluation
    """

    def __init__(self):
        self.tokenizer  = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
        self.preprocessor = TextPreprocessor()
        self.dataset    = LegalDataset()
        self.model: Optional[Model] = None
        self._is_trained = False
        self.label_map     = {t: i for i, t in enumerate(CASE_TYPES)}
        self.label_map_inv = {i: t for t, i in self.label_map.items()}

    # ──────────────────────────────────────────────
    # Model architecture
    # ──────────────────────────────────────────────

    def _build_model(self) -> Model:
        """
        BiLSTM architecture for long legal document classification.

        Input(512,)
          ↓
        Embedding(30000, 128)         — 30k vocab, 128-dim vectors
          ↓
        SpatialDropout1D(0.3)         — regularize embedding layer
          ↓
        Bidirectional(LSTM(128))      — reads both directions
          ↓
        BatchNormalization
          ↓
        Dropout(0.4)
          ↓
        Dense(64, relu)
          ↓
        Dropout(0.3)
          ↓
        Dense(7, softmax)             — 7 case types
        """
        inputs = Input(shape=(MAX_SEQ_LEN,), name="text_input")

        x = Embedding(
            input_dim=MAX_VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQ_LEN,
            name="word_embedding",
        )(inputs)

        x = SpatialDropout1D(0.3, name="spatial_dropout")(x)

        x = Bidirectional(
            LSTM(LSTM_UNITS, return_sequences=False),
            name="bilstm",
        )(x)

        x = BatchNormalization(name="batch_norm")(x)
        x = Dropout(DROPOUT_RATE, name="dropout_1")(x)

        x = Dense(64, activation="relu", name="dense_hidden")(x)
        x = Dropout(0.3, name="dropout_2")(x)

        outputs = Dense(
            len(CASE_TYPES),
            activation="softmax",
            name="output",
        )(x)

        model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_LegalClassifier_v2")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ──────────────────────────────────────────────
    # Data preparation (NO augmentation needed)
    # ──────────────────────────────────────────────

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from 1,414 real Supreme Court judgments.

        NO augmentation needed — real data is diverse enough.

        Steps:
          1. Load DataFrame (Pandas)
          2. Preprocess text (NLP pipeline)
          3. Tokenize → integer sequences
          4. Pad to MAX_SEQ_LEN (512)
          5. One-hot encode labels
        """
        print("[BiLSTM] Loading dataset...")
        df = self.dataset.get_cases_dataframe()

        print(f"[BiLSTM] Preprocessing {len(df)} texts (seq_len={MAX_SEQ_LEN})...")
        texts  = self.preprocessor.process_batch(df["full_text"].tolist())
        labels = df["case_type"].tolist()

        # Map unknown types to "civil" (default)
        labels = [l if l in self.label_map else "civil" for l in labels]

        print(f"[BiLSTM] Label distribution:")
        from collections import Counter
        dist = Counter(labels)
        for lbl, cnt in sorted(dist.items()):
            print(f"  {lbl:<16} {cnt}")

        # Fit tokenizer
        self.tokenizer.fit_on_texts(texts)
        vocab_size = min(len(self.tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
        print(f"[BiLSTM] Tokenizer vocabulary: {vocab_size:,} terms")

        # Convert to integer sequences
        sequences = self.tokenizer.texts_to_sequences(texts)

        # Pad to fixed length
        X = pad_sequences(
            sequences,
            maxlen=MAX_SEQ_LEN,
            padding="post",
            truncating="post",
        )

        # Encode labels
        y_int = np.array([self.label_map.get(l, 2) for l in labels])
        y     = to_categorical(y_int, num_classes=len(CASE_TYPES))

        print(f"[BiLSTM] X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    # ──────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────

    def train(self, epochs: int = EPOCHS, verbose: int = 1):
        """
        Train BiLSTM on 1,414 real Supreme Court judgments.

        With real data:
          - Larger model (30k vocab, 512 seq, 128 units)
          - No data augmentation
          - Checkpoint saves best model
          - Class weights for imbalanced data
        """
        X, y = self._prepare_data()

        # Compute class weights to handle imbalance
        label_counts = y.sum(axis=0)
        total        = label_counts.sum()
        class_weight = {
            i: total / (len(CASE_TYPES) * max(cnt, 1))
            for i, cnt in enumerate(label_counts)
        }
        print(f"\n[BiLSTM] Class weights: {class_weight}")

        # Build model
        self.model = self._build_model()
        print("\n[BiLSTM] Model Architecture:")
        self.model.summary()

        # Callbacks
        os.makedirs("models/saved/bilstm_classifier", exist_ok=True)
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath="models/saved/bilstm_classifier/best_model.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=0,
            ),
        ]

        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose,
        )

        self._is_trained = True

        best_acc     = max(history.history.get("accuracy", [0]))
        best_val_acc = max(history.history.get("val_accuracy", [0]))
        print(f"\n[BiLSTM] Training complete.")
        print(f"  Best train accuracy: {best_acc:.2%}")
        print(f"  Best val accuracy:   {best_val_acc:.2%}")

        return history

    # ──────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """Classify case text into one of 7 case types."""
        if not self._is_trained or self.model is None:
            return self._rule_based_fallback(text)

        processed = self.preprocessor.process(text)
        sequence  = self.tokenizer.texts_to_sequences([processed])
        padded    = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding="post")

        probs         = self.model.predict(padded, verbose=0)[0]
        predicted_idx = int(np.argmax(probs))
        predicted_type = self.label_map_inv.get(predicted_idx, "civil")
        confidence     = float(probs[predicted_idx])

        all_probs = {
            CASE_TYPES[i]: round(float(p), 3)
            for i, p in enumerate(probs)
        }

        return {
            "predicted_type":    predicted_type,
            "confidence":        round(confidence, 3),
            "confidence_pct":    f"{confidence:.1%}",
            "all_probabilities": all_probs,
        }

    def _rule_based_fallback(self, text: str) -> Dict:
        """Rule-based fallback (7 types including service and tax)."""
        text_lower = text.lower()
        scores = {
            "criminal":       sum(1 for k in ["murder", "302", "robbery", "fir", "accused", "narcotics", "terrorism"] if k in text_lower),
            "family":         sum(1 for k in ["divorce", "khul", "custody", "maintenance", "nikah", "marriage"] if k in text_lower),
            "civil":          sum(1 for k in ["contract", "breach", "property", "damages", "land", "civil appeal"] if k in text_lower),
            "corporate":      sum(1 for k in ["company", "director", "shareholder", "arbitration", "secp"] if k in text_lower),
            "constitutional": sum(1 for k in ["fundamental rights", "article 199", "writ", "habeas"] if k in text_lower),
            "service":        sum(1 for k in ["civil servant", "dismissal", "service tribunal", "government employee", "compulsory retirement"] if k in text_lower),
            "tax":            sum(1 for k in ["income tax", "sales tax", "fbr", "customs", "tax appeal"] if k in text_lower),
        }
        best  = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        conf  = scores[best] / total if total > 0 else 0.3
        probs = {k: round(v / total, 3) for k, v in scores.items()}
        return {
            "predicted_type":    best,
            "confidence":        round(conf, 3),
            "confidence_pct":    f"{conf:.1%}",
            "all_probabilities": probs,
        }

    def save(self, path: str = "models/saved/bilstm_classifier"):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras"))
        import pickle
        with open(os.path.join(path, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.tokenizer, f)
        print(f"[BiLSTM] Saved → {path}")

    def load(self, path: str = "models/saved/bilstm_classifier"):
        import pickle
        self.model = load_model(os.path.join(path, "model.keras"))
        with open(os.path.join(path, "tokenizer.pkl"), "rb") as f:
            self.tokenizer = pickle.load(f)
        self._is_trained = True
        print(f"[BiLSTM] Loaded from {path}")
