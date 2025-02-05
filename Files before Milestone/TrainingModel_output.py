from tensorflow import keras
model = keras.models.load_model('cnn_model.h5')
model.summary()
