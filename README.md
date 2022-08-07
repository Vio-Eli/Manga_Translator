
<p align="center">
  <img src="https://github.com/Vio-Eli/Manga_Translator/blob/main/images/logo.png" alt=""/>
</p>

## About
Input a manga panel and get the translated version! Currently, only does japanese to english.

The system, in its current state, can be divided into 6 main modules:
- Text Extraction
- Font Extraction (Planned)
- Image Cleaning
- Text Translation
- Overlaying Text + Font on Cleaned Image

Here is a very basic definiton of what each module does:
- Text Extraction: 
  - Uses CRAFT + Feature Extraction to extract text from an image
  - CRAFT is a CRNN that uses region score and affinity scores to determine where text is in an image
    - [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.pdf)
  - My Current Feature Extraction is based on HuggingFace's Vision Encoder Decoder [link](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder) (I've built my own but because of issues with training, currently I'm using kha-white's weights)

- Font Extraction (Planned):
  - A different type of Feature Extraction to extract the font from the text per box
  - Matches the font in the text box as closely as it can to over 50 different commonly used manga fonts
  - In the future, I want this to construct it's own fonts based on weights
 
- Image Cleaning:
  - Removes text from an image (easy to say, very hard to do)
  - The base version is just to white the box that the CRAFT model returns, however this will also remove parts of the image
  - The current plan is to carefully remove the exact text from the image, and then employ some sort of stitching algorithm to fix the image
  - Suggestions on how to accomplish this are appreciated
  
- Text Translation
  - Translates the japanese text read from the image.
  - Currently uses a python api of google translate. I plan to use a translation algorithm made by the university of tokyo for this purpose
  
- Overlaying Text + Font on the now cleaned image
  - Put the text, font, and everything together nicely and neatly
  - The Final Step
  

This is a project over many many months, and is currently not complete in the slightest.
Currently, I'm having an issue with training. My PC can only do about 3 TFLOPS, and because of that it takes months to train anything.

## Installation
Python 3.6+ is required. Python 3.9 is recommended.

Install requirements via `requirements.txt`
```py
pip install -r requirements.txt
```

Download the Craft model from [here](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view)
and into '/trainer/models'. Future versions will wget this automatically.

#### Optional Packages:
Pytorch for GPU, install as described [here](https://pytorch.org/get-started/locally/)

## Usage
Just edit and run main.py (argparse is coming)

## References
- CRAFT Github [link](https://github.com/clovaai/CRAFT-pytorch)
  - The Craft Paper [link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.pdf)
- Kha-White's Japanese Feature Extraction [link](https://github.com/kha-white/manga-ocr)
- Hugging Face's Vision Encoder Decoder [link](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder)
 
