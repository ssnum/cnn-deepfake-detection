#Data pre-processing
X_fake_list = []
X_real_list = []

for imagePath in FAKE:
  img = io.imread(imagePath)
  X_fake_list.append(img)

for imagePath in REAL:
  img = io.imread(imagePath)
  X_real_list.append(img)

X_fake = np.stack(X_fake_list, axis = 0)
X_real = np.stack(X_real_list, axis = 0)
print(X_fake.shape)
print(X_real.shape)

X = np.concatenate((X_fake, X_real), axis = 0)
print(X.shape)

y_fake = [0]*len(X_fake)
y_real = [1]*len(X_real)

y = np.array(y_fake + y_real)

#split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

#flatten matricies
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)








#KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 3) # try playing around with different values of n_neighbors

#fit the data
knn_model.fit(X_train_flattened, y_train)

#make predictions
predictions = knn_model.predict(X_test_flattened)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)








#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Train the model
rf.fit(X_train_flattened, y_train)

# Step 5: Make predictions
y_pred = rf.predict(X_test_flattened)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: ", accuracy)

# Additional evaluation metrics (optional)
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred))







#Logistic Regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flattened)
X_test_scaled = scaler.transform(X_test_flattened)

# Initialize Logistic Regression classifier
log_reg = LogisticRegression(random_state=42)

# Train the model
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
