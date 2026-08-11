from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import layers


X_train_cnn = X_train_cnn.reshape(-1, 32, 32, 3)  #reshape

X_test_cnn = X_test_cnn.reshape(-1, 32, 32, 3) #reshape

model = VGG16(include_top=False, input_shape=(32, 32, 3))
for layer in model.layers:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(model.output)
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)

model.compile(optimizer=SGD(lr=0.001, momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn))

score = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
