# Import necessary libraries
from sklearn import datasets
from sklearn.svm import SVC
from skimage import io
import numpy as np

# Load the data
digits = datasets.load_digits()
X = digits.images
y = digits.target

# Flatten the images
n_samples = len(X)
X = X.reshape((n_samples, -1))

# Create the classifier model
clf = SVC(gamma=0.001)
clf.fit(X, y)

def get_prediction(image_url):
    # Load the image from the URL
    image = io.imread(image_url)
    # Flatten the image
    image = image.reshape((1, -1))
    # Make a prediction
    prediction = clf.predict(image)
    return prediction
