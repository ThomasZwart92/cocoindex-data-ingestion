"""
Create a test image with text for OCR testing
"""
from PIL import Image, ImageDraw, ImageFont
import os

# Create an image with text
width, height = 800, 600
image = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image)

# Try to use a better font, fallback to default if not available
try:
    # Try Windows fonts
    font_large = ImageFont.truetype("arial.ttf", 48)
    font_medium = ImageFont.truetype("arial.ttf", 36)
    font_small = ImageFont.truetype("arial.ttf", 24)
except:
    # Use default font
    font_large = ImageFont.load_default()
    font_medium = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Add various text elements
draw.text((50, 50), "CocoIndex Document Processing", font=font_large, fill='black')
draw.text((50, 150), "Test Document for OCR", font=font_medium, fill='darkblue')

# Add paragraph text
paragraph = """This is a sample document created to test the image intelligence service.
It contains various text elements that should be detected by Google Vision OCR.
The service should extract this text and generate an AI caption describing
the content and layout of this image."""

y_pos = 250
for line in paragraph.split('\n'):
    draw.text((50, y_pos), line, font=font_small, fill='black')
    y_pos += 40

# Add some structured data
draw.text((50, 450), "Key Features:", font=font_medium, fill='darkgreen')
draw.text((80, 500), "• Optical Character Recognition (OCR)", font=font_small, fill='black')
draw.text((80, 530), "• AI-powered image captioning", font=font_small, fill='black')
draw.text((80, 560), "• Multi-language support", font=font_small, fill='black')

# Save the image
image.save('test_document.png')
print("Test image created: test_document.png")

# Also create a simple diagram-like image
diagram = Image.new('RGB', (600, 400), color='lightgray')
draw2 = ImageDraw.Draw(diagram)

# Draw some shapes
draw2.rectangle([50, 50, 250, 150], outline='blue', width=3)
draw2.text((100, 90), "Input Data", font=font_medium, fill='blue')

draw2.rectangle([350, 50, 550, 150], outline='green', width=3)
draw2.text((380, 90), "Processing", font=font_medium, fill='green')

draw2.rectangle([200, 250, 400, 350], outline='red', width=3)
draw2.text((250, 290), "Output", font=font_medium, fill='red')

# Draw arrows
draw2.line([250, 100, 350, 100], fill='black', width=2)
draw2.line([450, 150, 300, 250], fill='black', width=2)
draw2.line([200, 100, 100, 250], fill='black', width=2)

diagram.save('test_diagram.png')
print("Test diagram created: test_diagram.png")