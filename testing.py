# Author: Boden Kahn
# CSCI 403: Capstone
import os
import random
import multiprocessing
from super_gradients.training import Trainer, models
from super_gradients.training.metrics import DetectionMetrics_050
from collections import Counter
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_val
)
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)

def main():
    print(test(1))

# This method calculates the number of free parking spaces based on an image.
# The lotID parameter is an integer for the id of the desired parking lot and 
# the return value is an integer containing the number of free spaces in the 
# image as calculated by our computer vision model
def test(lotID = 1):
    # Specify parameters
    CHECKPOINT_DIR = '\\SpeedParkModel\\check_point'
    EXPERIMENT_NAME = 'SpeedPark'
    DATA_DIR = '/SpeedParkDatasets2' 
    TEST_IMAGES_DIR = 'test/images'
    TEST_LABELS_DIR = 'test/labels'
    MODEL_NAME = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    CLASSES = ['free_parking_space','not_free_parking_space']
    NUM_CLASSES = len(CLASSES)

    #dataloader params
    DATALOADER_PARAMS = {'batch_size': 4, 'num_workers': 2}

    test_data = coco_detection_yolo_format_val(
            dataset_params = {
                'data_dir': DATA_DIR,
                'images_dir': TEST_IMAGES_DIR,
                'labels_dir': TEST_LABELS_DIR,
                'classes': CLASSES
            },
            dataloader_params = DATALOADER_PARAMS
        )

    trainer = Trainer(experiment_name = EXPERIMENT_NAME, ckpt_root_dir = CHECKPOINT_DIR)

    # Get the best model from a specific training cycle
    best_model = models.get(MODEL_NAME,
                            num_classes = NUM_CLASSES,
                            checkpoint_path = os.path.abspath("C:\SpeedParkModel\check_point\SpeedPark\RUN_20250416_210632_718766\ckpt_best.pth"))

    # Print evaluation results
    print(trainer.test(model = best_model,
                test_loader = test_data,
                test_metrics_list = DetectionMetrics_050(score_thres = 0.1, 
                top_k_predictions = 300, 
                num_cls = NUM_CLASSES, 
                normalize_targets = True, 
                post_prediction_callback = PPYoloEPostPredictionCallback(
                    score_threshold = 0.01, 
                    nms_top_k = 1000, 
                    max_predictions = 300,
                    nms_threshold = 0.7))))

    # Determine which lot to look at and access the appropriate folder
    # Get the images to test on
    tpaths=[]
    imageNumber = random.randint(0, 2)
    directory = 'C:/tempTest/'
    directory += str(lotID) + '/'
    for dirname, _, filenames in os.walk(directory):
        filename = filenames[imageNumber]
        tpaths += [(os.path.join(dirname, filename))]

    # Save the results of the prediction from the best model
    result = best_model.predict(tpaths, conf = 0.6)#.show() #.images_predictions.save(output_folder = "C:\\tempTest\Outputs")

    labels = result.prediction.labels
    count = Counter(labels)
    free_spaces = count.get(0, 0)
    not_free_spaces = count.get(1, 0) # Currently unused

    return free_spaces
    

# Add multiprocessing support to stop freezing errors
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
