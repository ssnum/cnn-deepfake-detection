# Build the model
model_2 = Sequential()

# First convolutional layer
model_2.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same', input_shape=(32, 32, 3)))
model_2.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model_2.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))

# Add fully connected layer
model_2.add(Flatten())

model_2.summary()
