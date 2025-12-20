import numpy as np
import os
import sys

# Add parent directory to path to import from models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, EarlyStopping
except ImportError:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from KeyFrameDetection.models.cnn_lstm import build_cnn_lstm

def create_sequences(frames_data, sequence_length=10):
    """
    Create sequences from frames for LSTM training.
    For simplicity, we'll create synthetic labels (keyframes are frames with significant changes).
    """
    sequences = []
    labels = []
    
    # Use CNN feature extractor to get features (same architecture as in build_cnn_lstm)
    _, feature_extractor = build_cnn_lstm(input_shape=frames_data.shape[1:], sequence_length=sequence_length)
    
    # Extract features for all frames
    frame_features = feature_extractor.predict(frames_data, verbose=0)
    
    # Create sequences and labels based on frame differences
    for i in range(sequence_length, len(frame_features)):
        seq = frame_features[i-sequence_length:i]
        sequences.append(seq)
        
        # Simple heuristic: keyframe if there's significant change from previous frame
        if i > 0:
            change = np.mean(np.abs(frame_features[i] - frame_features[i-1]))
            label = 1 if change > np.percentile(np.abs(np.diff(frame_features, axis=0)), 75) else 0
        else:
            label = 0
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

def train_cnn_lstm(frames_data, epochs=50, batch_size=16, sequence_length=10, validation_split=0.2):
    """
    Train a CNN-LSTM model on frames data.
    
    Args:
        frames_data: numpy array of shape (num_frames, H, W, 3) - normalized to [0, 1]
        epochs: number of training epochs
        batch_size: batch size for training
        sequence_length: length of sequences for LSTM
        validation_split: fraction of data to use for validation
    
    Returns:
        Trained CNN-LSTM model
    """
    print(f"[INFO] Training CNN-LSTM on {len(frames_data)} frames")
    print(f"       Frame shape: {frames_data.shape}")
    print(f"       Sequence length: {sequence_length}")
    
    # Create sequences
    print("[INFO] Creating sequences...")
    sequences, labels = create_sequences(frames_data, sequence_length)
    print(f"       Created {len(sequences)} sequences")
    
    # Build model
    model, cnn_model = build_cnn_lstm(input_shape=frames_data.shape[1:], sequence_length=sequence_length)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create model directory
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'cnn_lstm_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        sequences,
        labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Save final models
    model.save(os.path.join(model_dir, 'cnn_lstm_final.h5'))
    cnn_model.save(os.path.join(model_dir, 'cnn_feature_extractor.h5'))
    print("[INFO] CNN-LSTM training completed!")
    
    return model

