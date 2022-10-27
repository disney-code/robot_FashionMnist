from tensorflow import keras
import tensorflow as tf
import numpy as np
from models import Lenet5, ConvNet_1
import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
from tensorflow.keras.callbacks import ModelCheckpoint
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        

# def load_fashion(path="./fashion.npz"):
#     f = np.load(path)
#     x_train, y_train = f['x_train'], f['y_train']
#     x_test, y_test = f['x_test'], f['y_test']
#     f.close()

#     x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#     x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#     x_train = x_train.astype('float32') / 255.
#     x_test = x_test.astype('float32') / 255.

#     y_train = keras.utils.to_categorical(y_train, 10)
#     y_test = keras.utils.to_categorical(y_test, 10)
    
#     return x_train, x_test, y_train, y_test


# path = "./fashion.npz"
(x_train, y_train), (x_vali, y_vali),(x_test, y_test) = load_dataset.load_Fmnist()
#x_train, x_test, y_train, y_test = load_fashion(path)

lenet5 = Lenet5()
lenet5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

save_dir = os.path.join(os.getcwd(), 'baseline')
model_name = 'FMNIST_10_lenet5_model.{epoch:03d}.h5' 
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')
callbacks=[checkpoint]

lenet5.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_vali,y_vali), callbacks=callbacks)

loss,accuracy=lenet5.evaluate(x_test, y_test)

print("Accuracy: ",accuracy)
# lenet5.save("./Lenet5_fashion.h5")
