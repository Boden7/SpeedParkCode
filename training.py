# Code sourced from https://www.labellerr.com/blog/building-parking-space-detection-system-pytorch-super-gradients/
# Dataset sourced from https://www.kaggle.com/datasets/duythanhng/parking-lot-database-for-yolo and https://www.kaggle.com/datasets/anandpanda3/parking-lot-object-detection-in-yolov8/data
# Adapted by Boden Kahn
# This file uses the pretrained model found on the website to create a tuned 
# model that searches for free or ocupied parking spaces in images of parking 
# lots. It is trained using the two datasets linked above to increase accuracy.
# 
# Run the following commands in the terminal before running the program
# pip install super-gradients
# pip install --upgrade pillow
# pip install torch
# pip install torch torchvision
# pip install --upgrade torchvision

import os
import multiprocessing
from torch import load
import torch.serialization
from numpy.core import multiarray
from numpy import ndarray
import numpy
from super_gradients.training import Trainer, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val
)
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)

def main():
    # Allow reconstruct ndarray, and dtype to be used globally
    torch.serialization.add_safe_globals({multiarray._reconstruct, ndarray, numpy.dtype})
    _original_torch_load = torch.load

    # Set weights only to false
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

    # Trainer params
    CHECKPOINT_DIR = '\\SpeedParkModel\\check_point'
    EXPERIMENT_NAME = 'SpeedPark'
    DATA_DIR = '/SpeedParkDatasets2' 
    TRAIN_IMAGES_DIR = 'train/images'
    TRAIN_LABELS_DIR = 'train/labels'
    VAL_IMAGES_DIR = 'valid/images'
    VAL_LABELS_DIR = 'valid/labels'
    TEST_IMAGES_DIR = 'test/images'
    TEST_LABELS_DIR = 'test/labels'

    CLASSES = ['free_parking_space','not_free_parking_space']
    NUM_CLASSES = len(CLASSES)

    # Dataloader params
    # num_workers at 8 because pc has 8 cores
    DATALOADER_PARAMS = {'batch_size': 16, 'num_workers': 8}

    # Model params
    MODEL_NAME = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    PRETRAINED_WEIGHTS = None # Using pretrained weights from a downloaded model file
    trainer = Trainer(experiment_name = EXPERIMENT_NAME, ckpt_root_dir = CHECKPOINT_DIR)

    # Specify data formats
    train_data = coco_detection_yolo_format_train(
    dataset_params = {
        'data_dir': DATA_DIR,
        'images_dir': TRAIN_IMAGES_DIR,
        'labels_dir': TRAIN_LABELS_DIR,
        'classes': CLASSES
    },
    dataloader_params = DATALOADER_PARAMS
)

    val_data = coco_detection_yolo_format_val(
        dataset_params = {
            'data_dir': DATA_DIR,
            'images_dir': VAL_IMAGES_DIR,
            'labels_dir': VAL_LABELS_DIR,
            'classes': CLASSES
        },
        dataloader_params = DATALOADER_PARAMS
    )

    test_data = coco_detection_yolo_format_val(
        dataset_params = {
            'data_dir': DATA_DIR,
            'images_dir': TEST_IMAGES_DIR,
            'labels_dir': TEST_LABELS_DIR,
            'classes': CLASSES
        },
        dataloader_params = DATALOADER_PARAMS
    )

    # Displays dataset, but images look overlapped or glitched.
    #train_data.dataset.plot()

    model = models.get(MODEL_NAME, num_classes = NUM_CLASSES, pretrained_weights = PRETRAINED_WEIGHTS)

    # Manually load the pretrained weights from the yolo_nas_l_coco.pth file
    checkpoint_path = r"C:\SpeedParkModel\PretrainedWeights\yolo_nas_l_coco.pth"
    checkpoint = load(checkpoint_path)
    state_dict = checkpoint['net']

    # Remove the 'cls_pred' weights from the loaded state_dict for all heads
    for head in ['head1', 'head2', 'head3']:
        state_dict.pop(f'heads.{head}.cls_pred.weight', None)
        state_dict.pop(f'heads.{head}.cls_pred.bias', None)

    # Load the state_dict into the model
    model.load_state_dict(state_dict, strict=False)

    # Replace the classifier layers to match number of classes (2)
    model.heads.head1.cls_pred = torch.nn.Conv2d(128, 2, kernel_size=1)
    model.heads.head2.cls_pred = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.heads.head3.cls_pred = torch.nn.Conv2d(512, 2, kernel_size=1)
                    
    train_params = {
        # ENABLING SILENT MODE
        "average_best_models": True,
        "warmup_mode": "linear_epoch_step",
        #'batch_accumulate': 2,
        "warmup_initial_lr": 1e-5,
        "lr_warmup_epochs": 2,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": False,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": 20,
        "mixed_precision": False, #mixed precision is not available for CPU 
        "loss": PPYoloELoss(
            use_static_assigner = False,
            num_classes = NUM_CLASSES,
            reg_max = 16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres = 0.1,
                top_k_predictions = 300,
                num_cls = NUM_CLASSES,
                normalize_targets = True,
                post_prediction_callback = PPYoloEPostPredictionCallback(
                    score_threshold = 0.01,
                    nms_top_k = 1000,
                    max_predictions = 300,
                    nms_threshold = 0.7
                )
            )
        ],
        # Watch for F1@0.50 to get the best recall and precision
        "metric_to_watch": 'F1@0.50',
        "save_checkpoints": True,
        "save_best": True,
        "save_weights_only": False,
    }
    
    # Train the model
    trainer.train(model = model, 
                training_params = train_params, 
                train_loader = train_data, 
                valid_loader = val_data)
    
    # Get the best model
    best_model = models.get(MODEL_NAME,
                            num_classes = NUM_CLASSES,
                            # Try it using the best checkpoint from a specific run
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
    
    # Get the images to test on
    tpaths=[]
    for dirname, _, filenames in os.walk('C:/SpeedParkDatasets2/test/images'):
        for filename in filenames:
            tpaths += [(os.path.join(dirname, filename))]
    print(len(tpaths))

    # Show the results from the best model
    best_model.predict(tpaths, conf = 0.6).show()

# Add multiprocessing support to stop freezing errors
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
