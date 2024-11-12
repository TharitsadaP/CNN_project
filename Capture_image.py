from controller import Robot, Camera
import cv2
import numpy as np

# Initialize the robot and set timestep
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Enable the camera
camera = robot.getDevice('camera')
camera.enable(timestep)

# Get the original width and height of the camera image
width = camera.getWidth()
height = camera.getHeight()

while robot.step(timestep) != -1:
    # Capture an image from the camera
    image = camera.getImage()

    # Convert the raw image data to a numpy array
    image_array = np.array([
        [
            [camera.imageGetRed(image, width, x, y),
             camera.imageGetGreen(image, width, x, y),
             camera.imageGetBlue(image, width, x, y)]
            for x in range(width)
        ] for y in range(height)
    ], dtype=np.uint8)

    # Convert the image array to BGR format for OpenCV
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Resize the image to 32x32
    resized_image = cv2.resize(bgr_image, (32, 32))

    # Save the resized image using OpenCV
    cv2.imwrite("captured_image_32x32.jpg", resized_image)
    print("Image saved as 'captured_imageCat2_32x32.jpg'")

    # Optionally, you can include a break condition to stop saving images
    break  # Remove or modify as needed
