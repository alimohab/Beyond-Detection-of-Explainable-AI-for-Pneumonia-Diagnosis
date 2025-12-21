import os
import sys

# Add parent directory to path to import from models
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from KeyFrameDetection.models.autoencoder import build_autoencoder

def train_autoencoder(train_data, img_size=(224, 224), epochs=40, batch_size=16, save_path="\models\autoencoder_best.h5",validation_split=0.1):

    """
    Train an autoencoder model on frames data.
    
    Args:
        train_data: numpy array of shape (num_frames, H, W, 3) - training data, normalized to [0, 1]
        test_data: numpy array of shape (num_frames, H, W, 3) - test data for validation, normalized to [0, 1]
                   If None, uses validation_split on train_data
        epochs: number of training epochs
        batch_size: batch size for training
        validation_split: fraction of data to use for validation (only used if test_data is None)
    
    Returns:
        Trained autoencoder model
    """
    
    # Build model
    model = build_autoencoder(input_shape=train_data.shape[1:])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
    
    # Create model directory
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    

    
    # Train the model (autoencoder learns to reconstruct input)
    history = model.fit(
        train_data,
        train_data, 
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Save final model
    print("[INFO] Autoencoder training completed!")
    print(f"[INFO] Best model saved at: {save_path}")
    
    return model, history

