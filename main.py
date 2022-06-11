from ocr.ocr import Reader
from PIL import Image, ImageDraw, ImageFont

image = './images/page0084.jpeg'

reader = Reader()

results = reader.read(image)

image = Image.open(image)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype('./images/NotoSansJP-Regular.otf', size=20)  # You can replace this with your own font

for result in results:
    box, text = result
    text.encode('utf-8', 'ignore')

    bo = [(x[0], x[1]) for x in box]
    bottom_left = (box[0][0], box[0][1])
    top_right = (box[2][0], box[2][1])

    draw.polygon(bo, outline=(255, 0, 0), width=3)  # Draws the box with red outline

    draw.text((box[2][0], box[0][1]), text, fill=(255, 0, 0), language='ja', font=font)  # Draws the text with red

image.show()


