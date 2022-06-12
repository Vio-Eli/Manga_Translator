from ocr.ocr import Reader
from PIL import Image, ImageDraw, ImageFont
import textwrap
# from googletrans import Translator

image = './images/page0084.jpeg'

reader = Reader()

results = reader.read(image)

image = Image.open(image)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype('./images/NotoSansJP-Regular.otf', size=15)  # You can replace this with your own font

# translator = Translator()

for box, text in results:

    # Deconstructing the box coords
    bo = [(x[0], x[1]) for x in box]  # Coords go top left, top right, bottom left, bottom right
    box_width = abs(bo[0][0] - bo[1][0])

    # Whiting out the box
    draw.polygon(bo, fill=(255, 255, 255, 0))

    # Translating the text
    # text.encode('utf-8', 'ignore')
    # text = translator.translate(text, src='ja', dest='en').text
    lines = textwrap.wrap(text, width=(int(box_width // font.getsize('あ')[0]) + 1))
    line_start = bo[0][1]  # Starting y coord of the first line

    for line in lines:
        line_width, line_height = font.getsize(line)
        draw.text(((box_width - line_width) / 2 + bo[0][0], line_start), line, font=font, fill='black')
        line_start += line_height

    draw.polygon(bo, outline=(255, 0, 0), width=3)  # Draws the box with red outline
    #
    # draw.text(bottom_left, text, fill=(0, 0, 0), language='ja', font=font)

image.show()


