model_4 = Sequential()

# First convolutional layer
model_4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model_4.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model_4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_4.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer
model_4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Fourth convolutional layer
model_4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# Add fully connected layer
model_4.add(Flatten())

model_4.summary()
