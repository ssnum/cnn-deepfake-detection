import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


df_results = pd.DataFrame(results)
df_results.to_csv('model_results.csv', index=False)
df_results


best_model_index = df_results['Validation Accuracy'].idxmax()
best_model = final_model[best_model_index]

# basically issue was that y_test_cnn was onehot encoded but y_pred was in binary so we had to convert one to the other
y_pred = best_model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)

y_test_classes = np.argmax(y_test_cnn, axis=1)


cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Best Model: 4 convolution [1024, 512, 256, 2] Confusion Matrix')
plt.show()


worst_model_index = df_results['Validation Accuracy'].idxmin()
worst_model = final_model[worst_model_index]

y_pred = worst_model.predict(X_test_cnn)
y_pred_classes_w = np.argmax(y_pred, axis=1)

y_test_classes_w = np.argmax(y_test_cnn, axis=1)


cm = confusion_matrix(y_test_classes_w, y_pred_classes_w)
plt.figure(figsize=(10, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Worst Model: 2 convolution [2] Confusion Matrix')
plt.show()



#print graphs
for model in final_model:
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
