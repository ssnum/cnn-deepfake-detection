import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load pre-trained DenseNet121 model (include_top=False to exclude top layers)
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Add a GlobalAveragePooling2D layer to reduce the dimensionality of the output from the base model
x = GlobalAveragePooling2D()(base_model.output)

# Connect the output of the base model to the Dense layer for predictions
# Use 2 units and softmax for multi-class classification
predictions = Dense(2, activation='softmax')(x)

# Create the Functional model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers from the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
# Use categorical_crossentropy for multi-class
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn))

# Evaluate the model
score = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
