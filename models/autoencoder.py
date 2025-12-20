"""
Autoencoder model for keyframe detection.
High reconstruction error indicates keyframes.
"""
try:
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
except ImportError:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

def build_autoencoder(input_shape=(224, 224, 3)):
    """
    Build an autoencoder model for keyframe detection.
    High reconstruction error indicates keyframes.
    
    Args:
        input_shape: Tuple of (height, width, channels)
    
    Returns:
        Compiled autoencoder model
    """
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 112x112
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  # 56x56
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # 28x28
    
    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)  # 56x56
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 112x112
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # 224x224
    
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = Model(inputs, decoded)
    return model
