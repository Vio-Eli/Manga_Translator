import os
import torch
from pathlib import Path

from .detector import get_detector, detect
from .recognizer import Recognizer
from .utils import reformat_img, get_image_list

from loguru import logger


class Reader(object):
    """ Main OCR Reader """
    def __init__(self,
                 gpu=True,
                 detector=True,
                 recognizer=True,
                 verbose=True,
                 quantize=True,
                 cudnn_benchmark=False):

        # Get dir of models, if it's not there, create it
        self.models_dir = Path('./trainer/models')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'
        self.verbose = verbose

        # Loading the detector (detects text boxes)
        if detector:
            detector_path = self.models_dir / 'craft_mlt_25k.pth'
            self.detector = get_detector(detector_path, self.device, quantize, cudnn_benchmark)
            logger.info(f'Successfully loaded detector on {self.device}')

        # Loading the recognizer (recognizes text from text boxes)
        if recognizer:
            self.recognizer = Recognizer(self.device)
            logger.info(f'Successfully loaded recognizer on {self.device}')

        logger.info(f'Reader initialized on {self.device}')

    def read(self, img):
        """ Reads the image and returns the text with their text box coords """
        img, img_cv_grey = reformat_img(img)  # Grab Image, convert to greyscale

        # Detect text boxes
        horizontal_list, free_list = detect(self.detector, self.device, img)

        if self.verbose:
            logger.debug(f'Detected {len(horizontal_list[0])} text boxes')

        horizontal_list, free_list = horizontal_list[0], free_list[0]  # We only want the first dimension

        # If there are no text boxes, make the horizontal list the entire image
        if (horizontal_list is None) and (free_list is None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        # Generating a list of cropped images from the text boxes
        image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey)

        if self.verbose:
            logger.debug(f'Generated {len(image_list)} crops with max width {max_width}')

        # A bit of formatting
        coords = [item[0] for item in image_list]
        img_list = [item[1] for item in image_list]

        # Recognize text from the cropped images
        text = [self.recognizer(img) for img in img_list]

        # Zipping the text and their coords
        result = [item for item in zip(coords, text)]
        return result













