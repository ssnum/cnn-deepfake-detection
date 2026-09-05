import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam

# ResNet50 backbone, pretrained on ImageNet
pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(32, 32, 3),
                                                  pooling='avg',
                                                  weights='imagenet')

# Partial fine-tuning: the first 7 residual blocks (conv1, conv2_block1-3,
# conv3_block1-4) act as general-purpose low/mid-level feature extractors and
# stay frozen. Block 8 onward (conv4_block1 through conv5_block3, the final
# convolutional stages) is unfrozen so the model can adapt its higher-level
# features to deepfake-specific patterns.
unfreeze_from = "conv4_block1"
unfreeze = False
for layer in pretrained_model.layers:
    if layer.name.startswith(unfreeze_from):
        unfreeze = True
    layer.trainable = unfreeze

resnet_model = Sequential([
    pretrained_model,
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='softmax'),
])

resnet_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = resnet_model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn))

# Evaluate the model
score = resnet_model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
