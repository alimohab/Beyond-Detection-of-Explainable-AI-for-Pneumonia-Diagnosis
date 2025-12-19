from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten,  Dense, TimeDistributed, LSTM, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
def build_keyframe_cnn(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)

    # ========== Encoder ==========
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)  # 112x112

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)  # 56x56

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2,2))(x)  # 28x28

    # ========== Decoder ==========
    x = Conv2D(128, (3,3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2,2))(x)  # 56x56

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)  # 112x112

    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)  # 224x224

    # ========== Key-frame Score ==========
    x = GlobalAveragePooling2D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)
    return model

model = build_keyframe_cnn()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()



model.save("Basic_Cnn_model.h5")
# preds = model.predict(frames)

# key_frames = [i for i, p in enumerate(preds) if p > 0.5]

# print("Key frame indices:", key_frames)