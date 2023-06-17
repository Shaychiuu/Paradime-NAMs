# Import the NAMClassifier class
import pandas as pd
from basicNAM import NAMClassifier

# Create an instance of NAMClassifier
classifier = NAMClassifier(hidden_dim=10, output_dim=1, num_layers=3)

# Load your dataset and preprocess it
data = pd.read_csv('C:\\Users\\isabe\\Documents\\AI studies\\6.Semester\\Bachelor Thesis\\Paradime-NAMs\\Datasets\\heart.csv')  # Replace with the path to your dataset
X = data.drop(columns=['target']).values
y = data['target'].values

# Fit the classifier to the data
classifier.fit(X, y)
predictions = classifier.predict(X)

# Make predictions using the trained classifier
# new_data = pd.read_csv('new_data.csv')  # Replace with the path to your new data
# X_new = new_data.values
# predictions = classifier.predict(X_new)
