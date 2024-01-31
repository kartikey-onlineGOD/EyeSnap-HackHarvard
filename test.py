import joblib
import cv2


# Load the model from the PKL file
loaded_model = joblib.load("logistic_regression_model.pkl")

# Load the image
path_bad = 'verify/badeye.png'
path_good = 'verify/goodeye.png'
image = cv2.imread(path_good)


# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image to the same dimensions used for training
resized_image = cv2.resize(gray_image, (64, 64))

# Flatten the image to match the input shape expected by the model
flattened_image = resized_image.reshape(1, -1)

prediction = loaded_model.predict(flattened_image)

if prediction == 1:
    print("The person shows high signs of Diabetic Retinopathy")
else:
    print("The person shows slight to no signs of Diabetic Retinopathy")
