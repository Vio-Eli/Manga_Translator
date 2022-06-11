import os
import torch
from pathlib import Path

from .detector import get_detector, detect
from .recognizer import Recognizer
from .utils import reformat_img, get_image_list

from logging import getLogger
LOGGER = getLogger(__name__)


class Reader(object):
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

        if detector:
            detector_path = self.models_dir / 'craft_mlt_25k.pth'
            self.detector = get_detector(detector_path, self.device, quantize, cudnn_benchmark)

        if recognizer:
            self.recognizer = Recognizer(self.device)

    def read(self, img):
        img, img_cv_grey = reformat_img(img)

        horizontal_list, free_list = detect(self.detector, self.device, img)

        horizontal_list, free_list = horizontal_list[0], free_list[0]

        if (horizontal_list is None) and (free_list is None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey)

        coords = [item[0] for item in image_list]
        img_list = [item[1] for item in image_list]
        text = [self.recognizer(img) for img in img_list]

        result = [item for item in zip(coords, text)]

        return result













