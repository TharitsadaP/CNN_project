from controller import Robot, Camera, Motor
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax


# Constants
INPUT_SHAPE = (32, 32, 3)  # Input shape expected by the CNN model
NUM_CLASSES = 10 
CLASS_NAME = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']           # CIFAR-10 dataset has 10 classes
#MODEL_PATH = "/Users/botan/Desktop/ARAP/Webot_CNN/controllers/Test_CNN/converted_model.h5 "  # Path to the saved CNN model
TIME_STEP = 64              # Time step for simulation
# Initialize robot, camera, and motors
robot = Robot()
camera = Camera('camera')
camera.enable(TIME_STEP)

# Motor setup
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))  # Setting to infinity for velocity control
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)           # Start with motors stopped
right_motor.setVelocity(0.0)

# Load the CNN model
cnn_model = load_model("/Users/botan/Desktop/ARAP/3Blocks_CNN/controllers/Test_CNN/converted_model.h5 ")

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))  # Resize to model input shape
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Main loop
while robot.step(TIME_STEP) != -1:
    # Capture image from the camera
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))[:, :, :3]  # Get RGB only
    processed_image = preprocess_image(image)
    
    # Make prediction
    # Make prediction
    prediction = cnn_model.predict(processed_image)
    prediction_probabilities = softmax(prediction).numpy()  # Convert logits to probabilities if necessary
    predicted_class = np.argmax(prediction_probabilities)
    confidence = np.max(prediction_probabilities)
# Get the name of the predicted class
    predicted_class_name = CLASS_NAME[predicted_class]

# Print the predicted class name and confidence
    print(f"Predicted Class: {predicted_class_name}, Confidence: {confidence}")

    #prediction = cnn_model.predict(processed_image)
    #predicted_class = np.argmax(prediction)  # Get the class with the highest probability
    #confidence = np.max(prediction)          # Get the confidence of the prediction

    # Print the predicted class and confidence
    #print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

    # Simple movement logic based on prediction
    if predicted_class == 0:  # Example: Move forward if class 0 is detected
        left_motor.setVelocity(3.0)
        right_motor.setVelocity(3.0)
    elif predicted_class == 1:  # Example: Turn left if class 1 is detected
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(3.0)
    elif predicted_class == 2:  # Example: Turn right if class 2 is detected
        left_motor.setVelocity(3.0)
        right_motor.setVelocity(0.0)
    else:  # Stop for other classes
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
