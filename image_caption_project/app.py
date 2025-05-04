import os
from flask import Flask, request, render_template
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            image = Image.open(path).convert('RGB')

            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            image_path = path

    return render_template('index.html', caption=caption, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
