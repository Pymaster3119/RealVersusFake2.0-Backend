import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from io import BytesIO
from transformers import pipeline
import base64
import torch

pipe = pipeline(
    'image-classification', 
    model="date3k2/vit-real-fake-classification-v3", 
    device=0, 
    torch_dtype=torch.float16
)

app = Flask(__name__)
CORS(app, resources={r"/check-image": {"origins": "*", "methods": ["OPTIONS", "POST"], "headers": ["Content-Type"]}})

def process_and_classify(image):
    processed = image.convert("RGB").resize((256, 256)).convert("L")
    image.close()
    result = pipe(processed)
    return result

@app.route('/check-image', methods=['POST'])
def check_image():
    image_url = request.json.get("image_url")
    if not image_url.startswith("data:"):
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            result = process_and_classify(image)
        else:
            return jsonify({'error': 'Unable to fetch image'}), 400
    else:
        _, base64_data = image_url.split(',', 1)
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
        result = process_and_classify(image)

    if result[0]["label"] == "Real":
        return jsonify({'is_ai_generated': False, "score": result[0]["score"]})
    else:
        return jsonify({'is_ai_generated': True, "score": result[0]["score"]})

if __name__ == '__main__':
    app.run(port=5000)