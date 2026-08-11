model_nodes = [[2], [512,2], [512,512,2], [1024,512,256,2]]
convolution_models = [model_2,model_3,model_4]
final_model = []
model_names = []
convolution_names = ['2 convolution', '3 convolution', '4 convolution']


#this is where i create 12 models
for idx, cmodel in enumerate(convolution_models):
  for nodes in model_nodes:
      model_names.append(convolution_names[idx] + " " + str(nodes))
      current_model = Sequential()
      current_model.add(cmodel)
      #print(nodes)
      for node in nodes:
          if node == nodes[-1]:
            current_model.add(Dense(node, activation='softmax'))
          else:
            current_model.add(Dense(node, activation='relu'))

      final_model.append(current_model)

      print("\n\n\n\nNodes Structure ", nodes)
      print(current_model.summary())






#training each model
results = []
for idx, model in enumerate(final_model):
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  history = model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn,y_test_cnn))
  train_acc = history.history['accuracy'][-1]
  val_acc = history.history['val_accuracy'][-1]
  train_loss = history.history['loss'][-1]
  val_loss = history.history['val_loss'][-1]
  results.append({
        'Model Name': model_names[idx],
        'Train Accuracy': train_acc,
        'Validation Accuracy': val_acc,
        'Train Loss': train_loss,
        'Validation Loss': val_loss
    })

  #plot
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)

  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')

  # Plot training & validation loss values
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.savefig(model_names[final_model.index(model)] + '.png')

  plt.show()


  #Evaluate the model
  print(model_names[final_model.index(model)])
  score = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])


