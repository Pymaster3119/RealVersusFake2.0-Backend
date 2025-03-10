import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from io import BytesIO
#Model loaded from HuggingFace
from transformers import pipeline
import base64

pipe = pipeline('image-classification', model="date3k2/vit-real-fake-classification-v3", device=0)

app = Flask(__name__)
CORS(app, resources={r"/check-image": {"origins": "*", "methods": ["OPTIONS", "POST"], "headers": ["Content-Type"]}})

@app.route('/check-image', methods=['POST'])
def check_image():
    image_url = request.json.get("image_url")
    #If it is actually a URL
    if not image_url.startswith("data:"):
        response = requests.get(image_url)
        if response.status_code == 200:
            print("Response recieved ")
            image = Image.open(BytesIO(response.content))
            image = image.resize((256,256)).convert("RGB").convert("L")
            result = pipe(image)
            print(result)
            if result[0]["label"] == "Real":
                return jsonify({'is_ai_generated': False, "score": result[0]["score"]})
            else:
                return jsonify({'is_ai_generated': True, "score": result[0]["score"]})
    #If it is a URI
    else:
        _, base64_data = image_url.split(',', 1)
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
        image = image.resize((256,256)).convert("RGB").convert("L")
        result = pipe(image)
        print(result)
        if result[0]["label"] == "Real":
            return jsonify({'is_ai_generated': False, "score": result[0]["score"]})
        else:
            return jsonify({'is_ai_generated': True, "score": result[0]["score"]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
