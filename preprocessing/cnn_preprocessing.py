y_train_cnn = []
y_test_cnn = []
for i in y_train:
  if i == 0:
    y_train_cnn.append(np.array([1,0]))
  else:
    y_train_cnn.append(np.array([0,1]))

for i in y_test:
  if i == 0:
    y_test_cnn.append(np.array([1,0]))
  else:
    y_test_cnn.append(np.array([0,1]))

y_train_cnn = np.array(y_train_cnn)
y_test_cnn = np.array(y_test_cnn)


X_train_cnn = X_train_flattened.astype(float)
X_test_cnn = X_test_flattened.astype(float)

print(X_test_cnn.shape)

X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], 32, 32, 3)
X_test_cnn = X_test_cnn.reshape(X_test_cnn.shape[0], 32, 32, 3)

print(X_train_cnn.shape)
print(X_test_cnn.shape)
print(y_train_cnn.shape)
print(y_test_cnn.shape)
