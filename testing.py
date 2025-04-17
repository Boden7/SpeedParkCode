import os
from super_gradients.training import models

# Specify parameters
MODEL_NAME = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
CLASSES = ['free_parking_space','not_free_parking_space']
NUM_CLASSES = len(CLASSES)

# Get the best model from a specific training cycle
best_model = models.get(MODEL_NAME,
                        num_classes = NUM_CLASSES,
                        checkpoint_path = os.path.abspath("C:\SpeedParkModel\check_point\SpeedPark\RUN_20250415_110116_528044\ckpt_best.pth"))

# Get the images to test on
tpaths=[]
for dirname, _, filenames in os.walk('C:/tempTest'):
    for filename in filenames:
        tpaths += [(os.path.join(dirname, filename))]
print(len(tpaths))

# Save the results of the predictions from the best model
best_model.predict(tpaths, conf = 0.35).images_predictions.save(output_folder = "C:\\tempTest\Outputs")