from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import csv
from glob import glob
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix

# This is the size of our encoded representations
encoding_dim = 33  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(shape=(333,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(333, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

PATH = "./Dataset/Normal"
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]

df = pd.DataFrame()
for name in all_csv_files:
    df1 = pd.read_csv(name)
    new_dict = dict()
    try:
        for index, row in df1.iterrows():
            new_dict[row['syscall']] = row['count']
    except:
        print(name)
        raise KeyError
    df =  df.append(new_dict, ignore_index=True)

for i in range(333):
    if i not in df:
        df[i] = [0 for j in range(len(df))]
df = df.fillna(0)
df = df.sample(frac=1)
y = df[1]

X_train, X_test, Y_train, Y_test = train_test_split(df, y, test_size=0.1)
print(X_train.shape)
print(X_test.shape)

# from tensorflow.keras.datasets import mnist
# import numpy as np
# (x_train, _), (x_test, _) = mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)

autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=256,
                shuffle=True)

# # Encode and decode some digits
# # Note that we take them from the *test* set
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)
# print(decoded_imgs)
a = np.power(X_test - decoded_imgs, 2)
mse = np.mean(a, axis=1)
print("power : ",a)
with open("mse.txt","w") as f:
    for i in mse:
        print(i,file=f)
print(type(mse))
mse_max = max(mse)
# # Use Matplotlib (don't ask)
# import matplotlib.pyplot as plt

# n = 10  # How many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # Display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # Display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.savefig("mygraph.png")

PATH = "./Dataset/Attack"
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]

df = pd.DataFrame()
for name in all_csv_files:
    df1 = pd.read_csv(name)
    new_dict = dict()
    try:
        for index, row in df1.iterrows():
            new_dict[row['syscall']] = row['count']
    except:
        print(name)
        raise KeyError
    df =  df.append(new_dict, ignore_index=True)

for i in range(333):
    if i not in df:
        df[i] = [0 for j in range(len(df))]
df = df.fillna(0)
df = df.sample(frac=1)
y = df[1]

X_train, X_test, Y_train, Y_test = train_test_split(df, y, test_size=0.01)


encoded_imgs = encoder.predict(X_train)
decoded_imgs = decoder.predict(encoded_imgs)
# print(decoded_imgs)
a = np.power(X_train - decoded_imgs, 2)
mse = np.mean(a, axis=1)
f = open("mse.txt","a")
print("---------------",file=f)
print(mse,file=f)
print("---------------",file=f)
for i in mse:
    if i>mse_max:
        print("Attack detected")
    print(i,file=f)
f.close()