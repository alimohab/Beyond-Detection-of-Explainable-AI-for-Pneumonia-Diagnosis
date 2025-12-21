"""
CNN-LSTM model for keyframe detection.
Uses CNN to extract features from frames, then LSTM to capture temporal patterns.
"""
try:
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, TimeDistributed, LSTM, Dense, GlobalAveragePooling2D
except ImportError:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, TimeDistributed, LSTM, Dense, GlobalAveragePooling2D

def build_cnn_lstm(input_shape=(224, 224, 3), sequence_length=10):
    """
    Build a CNN-LSTM model for keyframe detection.
    Uses CNN to extract features from frames, then LSTM to capture temporal patterns.
    
    Args:
        input_shape: Tuple of (height, width, channels) for single frame
        sequence_length: Length of sequences for LSTM
    
    Returns:
        Tuple of (end-to-end CNN-LSTM model, CNN feature extractor model for inference)
    """
    # CNN feature extractor (will be shared and trained end-to-end)
    cnn_input = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input)
    x = MaxPooling2D((2, 2))(x)  # 112x112
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)  # 56x56
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)  # Flatten to feature vector (128 dims)
    
    # CNN model for feature extraction (used separately for inference)
    cnn_model = Model(cnn_input, x, name='cnn_feature_extractor')
    

    sequence_input = Input(shape=(sequence_length, input_shape[0], input_shape[1], input_shape[2]))
    

    cnn_features = TimeDistributed(cnn_model)(sequence_input)
    
    # LSTM layers that process the sequence of CNN features
    lstm_out = LSTM(64, return_sequences=True)(cnn_features)
    lstm_out = LSTM(32, return_sequences=False)(lstm_out)
    
    output = Dense(1, activation='sigmoid')(lstm_out)
    
    model = Model(sequence_input, output, name='cnn_lstm_end_to_end')
    
    return model, cnn_model
