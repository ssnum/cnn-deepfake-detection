# Build the model
model_3 = Sequential()

# Add convolutional layer
model_3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model_3.add(MaxPooling2D(pool_size=(2, 2)))

# Add another convolutional layer
model_3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))

# Add another convolutional layer
model_3.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))

# Add fully connected layer
model_3.add(Flatten())

model_3.summary()
