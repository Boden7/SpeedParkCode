# Author: Boden Kahn
# CSCI 403: Capstone

import os
import random
import multiprocessing
from super_gradients.training import models
import glob
from collections import Counter

test_mode = True

def main():
    print(getSpaceCount(1))

# This method calculates the number of free and taken parking spaces in a 
# parking lot based on an image. It selects a random image from the folder
# representing a parking lot, which is named according to the lotID number.
#
# Boden Kahn
# 
# @param lotID  The id number of the lot to analyze.
# @return       The a list containing the number of free spaces in the image as
#               calculated by the machine learning model and the number of 
#               unavailable spaces as calculated by the model.
def getSpaceCount(lotID = 1):
    # Specify parameters
    MODEL_NAME = 'yolo_nas_l'  # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    CLASSES = ['free_parking_space', 'not_free_parking_space']
    NUM_CLASSES = len(CLASSES)

    # Load the model
    best_model = models.get(MODEL_NAME,
                            num_classes = NUM_CLASSES,
                            checkpoint_path = os.path.abspath("C:/SpeedParkModel/check_point/SpeedPark/RUN_20250427_082814_640062/ckpt_best.pth"))# Replace with current best model

    # Find available test images in the lot's folder
    directory = f"C:/tempTest/{lotID}/"
    all_images = glob.glob(os.path.join(directory, "**", "*.jpg"), recursive = True)

    if not all_images:
        print(f"No images found in {directory}")
        return -1

    # Pick a random valid image
    chosen_image = random.choice(all_images)
    print(f"Selected image: {chosen_image}")

    # Predict using the model
    result = best_model.predict(chosen_image, conf = 0.6)

    if not result:  # If no results, exit
        print("No predictions were made.")
        return -1

    # Visualize results if in testing mode
    if(test_mode):
        result.show()

    # Get the labels from the prediction
    labels = result.prediction.labels

    # Count occurrences of each label
    count = Counter(labels)
    free_spaces = count.get(0, 0)
    not_free_spaces = count.get(1, 0)

    # Return the number of available and unavailable spaces
    return [free_spaces, not_free_spaces]


# Add multiprocessing support to stop freezing errors
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
