import re
from pathlib import Path

import jaconv
import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel


class Recognizer:
    """ Recognize text from image """

    def __init__(self, device):
        self.model = VisionEncoderDecoderModel.from_pretrained('kha-white/manga-ocr-base')
        self.tokenizer = AutoTokenizer.from_pretrained('kha-white/manga-ocr-base')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('kha-white/manga-ocr-base')

        if torch.cuda.is_available() and device == 'cuda':
            self.model.cuda()

        self.model.eval()

    def __call__(self, img):

        img = Image.fromarray(np.uint8(img)).convert('L').convert('RGB')

        x = self._preprocess(img)
        x = self.model.generate(x[None].to(self.model.device))[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x

    def _preprocess(self, img):
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text
