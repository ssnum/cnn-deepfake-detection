from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize the model
model = Sequential()

# Add convolutional layers with Batch Normalization
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add more convolutional layers with padding to maintain spatial dimensions
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')) # Add padding
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a dropout layer for regularization
model.add(Dropout(0.5))

# Flatten the output of the last convolutional layer
model.add(Flatten())

# Add fully connected layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

# Output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Train the model using augmented data
batch_size = 32
history = model.fit(datagen.flow(X_train_cnn, y_train_cnn, batch_size=batch_size),
                    steps_per_epoch=len(X_train_cnn) / batch_size,
                    epochs=20,
                    validation_data=(X_test_cnn, y_test_cnn))

# Evaluate the model
score = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
