#preprocessing for data visualization 
import random

X = np.array(random.sample(avg_brightness_values_fake, k =20000) + avg_brightness_values_real).reshape(-1, 1)

# Create the labels (0 for fake, 1 for real)
y_fake = [0] * 20000
y_real = [1] * len(avg_brightness_values_real)

# Combine the labels into a single list (label vector y)
y = np.array(y_fake + y_real)

#split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

#flatten matricies
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)



#KNN model: This model works by measuring the distance between data points given in the X matrix and then grouping alike data together. 
#Depending on an integer we assigin, the algorithm will look for n amount of closest data points and whichever group the currently tested data point is closest to, will be the assigned result. 
#We can change the amount of "neighbors" the algorithm measures by increasing the number of data points the algorithm checks to assign the result.
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 2) # try playing around with different values of n_neighbors

#fit the data
knn_model.fit(X_train_flattened, y_train)

#make predictions
predictions = knn_model.predict(X_test_flattened)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)





#Random Forest Classifier: This is a classifying strategy that is based on the idea of Decision Trees. 
#The Random Forest Classifier relies on the concept of splitting the dataset into different subsets, and narrowing each subset down to a result using numerous different Decision Trees. 
#The average result of all the decision trees is used to give the final prediction, which helps improve accuracy by reducing the impact of potential errors or biases within individual trees.

#Note that the reason for the low accuracy for real images is due to the overwhelming number of fake (60000) vs (20000) real images, so the decision trees had majority fake images when they were being trained.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Train the model
rf.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = rf.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: ", accuracy)

# Additional evaluation metrics (optional)
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Bootstrapper Method: Bootstrapper involves generating multiple subsets of the original dataset by randomly sampling with replacement. 
#Each subset, known as a bootstrap sample, is used to train a separate model (typically decision trees in the case of Random Forests). 
#The predictions from all these models are then aggregated to produce a final prediction. 
#This aggregation can be done through averaging for regression tasks or majority voting for classification tasks.
num_bootstraps = 20
accuracies = []

for _ in range(num_bootstraps):
    # Generate bootstrap samples
    indices = np.random.choice(len(X), len(X), replace=True)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]

    # Split the bootstrap data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_bootstrap, y_bootstrap, test_size=0.2, random_state=42)

    # Initialize the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Calculate mean of accuracies
mean_accuracy = np.mean(accuracies)

print("Mean Accuracy: ", mean_accuracy)




#Logistic Regression: This method uses a mathematical approach and compares the predicted value from the mathematical equation to the actual value (this difference is considered the error). 
#It uses the logistic equation, or the sigmoid equation, to estimate the probability of a certain brightness value (since my inputs are the brightnesses) and then to assign that to either 0 or 1 (fake or real).
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Logistic Regression classifier
log_reg = LogisticRegression(random_state=42)

# Train the model
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)





#Support Vector Classification: This is a method that tries to separate the data points into two groups (in my case, fake and real). 
#The computer will try to draw a linear model (line) between the two groups, graphed on a hyperplane, such that the margin (distance between points in different groups) is maximised. 
#By doing this, the computer is then able to assign everything in the test data to a certain side of the hyperplane depending on the model it created when using the train data.
from sklearn.svm import SVC

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Support Vector Classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Train the model
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)






