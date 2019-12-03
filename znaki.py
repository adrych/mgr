#import mrcnn.model as modellib
#import mrcnn.utils as utils
#from mrcnn.config import Config




"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug

MGR_RCNN = '/content/gdrive/MGR_RCNN/COCO/'
# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/znaki"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Import Mask RCNN
sys.path.append(ROOT_DIR)

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(MGR_RCNN, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(MGR_RCNN, "logs")

############################################################
#  Configurations
############################################################


class ZnakiConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "znaki"
    BATCH_SIZE = 1 # DAREK exp
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 43  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 125 #org 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (21, 21)
    IMAGE_MIN_DIM = 832
    IMAGE_MAX_DIM = 1408

    TRAIN_ROIS_PER_IMAGE = 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

############################################################
#  Dataset
############################################################

class ZnakiDataset(utils.Dataset):

    def load_znaki(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        #self.add_class("znaki", 1, "balloon")
        self.add_class("znaki", 0, "speed limit 20 (prohibitory)")
        self.add_class("znaki", 1, "speed limit 30 (prohibitory)")
        self.add_class("znaki", 2, "speed limit 50 (prohibitory)")
        self.add_class("znaki", 3, "speed limit 60 (prohibitory)")
        self.add_class("znaki", 4, "speed limit 70 (prohibitory)")
        self.add_class("znaki", 5, "speed limit 80 (prohibitory)")
        self.add_class("znaki", 6, "restriction ends 80 (other)")
        self.add_class("znaki", 7, "speed limit 100 (prohibitory)")
        self.add_class("znaki", 8, "speed limit 120 (prohibitory)")
        self.add_class("znaki", 9, "no overtaking (prohibitory)")
        self.add_class("znaki", 10, "no overtaking (trucks) (prohibitory)")
        self.add_class("znaki", 11, "priority at next intersection (danger)")
        self.add_class("znaki", 12, "priority road (other)")
        self.add_class("znaki", 13, "give way (other)")
        self.add_class("znaki", 14, "stop (other)")
        self.add_class("znaki", 15, "no traffic both ways (prohibitory)")
        self.add_class("znaki", 16, "no trucks (prohibitory)")
        self.add_class("znaki", 17, "no entry (other)")
        self.add_class("znaki", 18, "danger (danger)")
        self.add_class("znaki", 19, "bend left (danger)")
        self.add_class("znaki", 20, "bend right (danger)")
        self.add_class("znaki", 21, "bend (danger)")
        self.add_class("znaki", 22, "uneven road (danger)")
        self.add_class("znaki", 23, "slippery road (danger)")
        self.add_class("znaki", 24, "road narrows (danger)")
        self.add_class("znaki", 25, "construction (danger)")
        self.add_class("znaki", 26, "traffic signal (danger)")
        self.add_class("znaki", 27, "pedestrian crossing (danger)")
        self.add_class("znaki", 28, "school crossing (danger)")
        self.add_class("znaki", 29, "cycles crossing (danger)")
        self.add_class("znaki", 30, "snow (danger)")
        self.add_class("znaki", 31, "animals (danger)")
        self.add_class("znaki", 32, "restriction ends (other)")
        self.add_class("znaki", 33, "go right (mandatory)")
        self.add_class("znaki", 34, "go left (mandatory)")
        self.add_class("znaki", 35, "go straight (mandatory)")
        self.add_class("znaki", 36, "go right or straight (mandatory)")
        self.add_class("znaki", 37, "go left or straight (mandatory)")
        self.add_class("znaki", 38, "keep right (mandatory)")
        self.add_class("znaki", 39, "keep left (mandatory)")
        self.add_class("znaki", 40, "roundabout (mandatory)")
        self.add_class("znaki", 41, "restriction ends (overtaking) (other)")
        self.add_class("znaki", 42, "restriction ends (overtaking (trucks)) (other)")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(
            open(os.path.join(dataset_dir, "ann2.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] #.values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "znaki",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        # If not a znaki dataset image, delegate to parent class.
        instance_masks = []
        class_ids = []
        image_info = self.image_info[image_id]
        #print(image_info.__dic__)
               
        if image_info["source"] != "znaki":
            return super(self.__class__, self).load_mask(image_id)
        
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            #print('dostalem p: %s' % p)
            rr, cc = skimage.draw.rectangle((p['y'], p['x']), (p['y']+p['width'], p['height']+p['x']))
            #print('get values %s %s' % (rr,cc))
            mask[rr, cc, i] = 1 #p["class_id"]
            #mask[rr, cc] = 1 #p["class_id"]
            m = mask
            #print(m)
            #class_ids = np.array(p["class_id"], dtype=np.int32)
            instance_masks.append(m)
            class_ids.append(p["class_id"])
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #return mask.astype(np.bool), class_ids #np.ones([mask.shape[-1]], dtype=np.int32)
        if class_ids:
            #mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
            #return mask.astype(np.bool), #np.ones([mask_array.shape[-1]], dtype=np.int32)
        else:
            # Call super class to return an empty mask
            return super(ZnakiDataset, self).load_mask(image_id)


    def load_mask2(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "znaki":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in info])
        return mask, class_ids.astype(np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "znaki":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ZnakiDataset()
    dataset_train.load_znaki(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ZnakiDataset()
    dataset_val.load_znaki(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
            datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(
            datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect znaki.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ZnakiConfig()
    else:
        class InferenceConfig(ZnakiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        #train(model)
    
        # Training dataset.
        dataset_train = ZnakiDataset()
        dataset_train.load_znaki(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = ZnakiDataset()
        dataset_val.load_znaki(args.dataset, "val")
        dataset_val.prepare()
        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='heads',
            augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers='4+',
            augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=160,
            layers='all',
            augmentation=augmentation)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
