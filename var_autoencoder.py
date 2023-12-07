import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda
from keras.backend import random_normal, mean, exp, square
from keras.losses import binary_crossentropy
from keras.models import Model
import matplotlib.pyplot as plt

# MNIST data
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain = xTrain/ 255.
xTest = xTest/ 255.
xTrain = xTrain.reshape((len(xTrain), np.prod(xTrain.shape[1:])))
xTest = xTest.reshape((len(xTest), np.prod(xTest.shape[1:])))

pixDims = 784
intLayer1 = 256
intLayer2 = 128
lat = 2

#encoder
encInput = Input(shape=(pixDims,))
h1 =Dense(intLayer1, activation='relu')(encInput)
h2 =Dense(intLayer2, activation='relu')(h1)
zm = Dense(lat)(h2)

zvlog = Dense(lat)(h2)

# Z sample function
def getz(inputs):
    zm =inputs[0]
    zvlog =inputs[1]
    eps = random_normal(shape=(tf.shape(zm)[0], tf.shape(zm)[1]))
    return zm + tf.exp(0.5 * zvlog) * eps

z = Lambda(getz, output_shape=(lat,))([zm, zvlog])

# Define the decoder
dec_h1 = Dense(intLayer2, activation='relu')
dec_h2 = Dense(intLayer1, activation='relu')
decoder_out = Dense(pixDims, activation='sigmoid')
dec_layer1 = dec_h1(z)
dec_layer2 = dec_h2(dec_layer1)
x_decoded = decoder_out(dec_layer2)

# Set up VAE model
vae = Model(encInput, x_decoded)

#L loss
total_loss = mean(pixDims * binary_crossentropy(encInput, x_decoded) - 0.5 * mean(1 + zvlog - square(zm) - exp(zvlog), axis=-1))

# Train model
vae.add_loss(total_loss)
vae.compile(optimizer='adam')
vae.fit(xTrain, None, epochs=10, batch_size=64, validation_data=(xTest, None))

# Decoder model for the 2D latent space visualization
decInput = Input(shape=(lat,))
_dec_h1 = dec_h1(decInput)
_dec_h2 = dec_h2(_dec_h1)
_decode_out = decoder_out(_dec_h2)
decoder_model = Model(decInput, _decode_out)

# Encoder model definition
encoder = Model(encInput, [zm, zvlog, z])

# Visualize results in latent space
mu, _, _ = encoder.predict(xTest)
plt.figure(figsize=(10, 10))
plt.scatter(mu[:, 0], mu[:, 1], c=yTest, cmap='brg')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.colorbar()
plt.savefig("latentSpace.png")  # Save the scatter plot
plt.show()

# Visualize images from the latent space
sample_vector = np.array([[-3,1]])
decoded_example = decoder_model.predict(sample_vector)
decoded_example_reshaped = decoded_example.reshape(28, 28)
plt.imshow(decoded_example_reshaped)
plt.savefig("digitSample.png")  # Save the sample digit
plt.show()

# 2D manifold of the digits
n = 30  # grid size
figure = np.zeros((28 * n, 28 * n))
grid_x = np.linspace(-2, 2, n)  
grid_y = np.linspace(-2, 2, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder_model.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.axis('off')
plt.savefig("numberGrid.png") 
plt.show()
