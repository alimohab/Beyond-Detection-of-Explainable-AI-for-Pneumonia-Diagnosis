import numpy as np
import os
import sys

# Add parent directory to path to import from models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, EarlyStopping
except ImportError:
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from KeyFrameDetection.models.autoencoder import build_autoencoder

def train_autoencoder(frames_data, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train an autoencoder model on frames data.
    
    Args:
        frames_data: numpy array of shape (num_frames, H, W, 3) - normalized to [0, 1]
        epochs: number of training epochs
        batch_size: batch size for training
        validation_split: fraction of data to use for validation
    
    Returns:
        Trained autoencoder model
    """
    print(f"[INFO] Training autoencoder on {len(frames_data)} frames")
    print(f"       Frame shape: {frames_data.shape}")
    
    # Build model
    model = build_autoencoder(input_shape=frames_data.shape[1:])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Create model directory
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'autoencoder_best.h5'),
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
    
    # Train the model (autoencoder learns to reconstruct input)
    history = model.fit(
        frames_data,
        frames_data,  # Autoencoder: input = target
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_dir, 'autoencoder_final.h5'))
    print("[INFO] Autoencoder training completed!")
    
    return model

