from flask import Flask, renderpage, request
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

model = joblib.load("savedmodel.pth")

@app.route('/')
def home():
    return '''<h2>Upload an image of size 64x64 grayscale</h2>
              <form method="POST" action="/predict" enctype="multipart/form-data">
                 <input type="file" name="file"/>
                 <input type="submit"/>
              </form>'''

@app.route('/predict', methods=['POST'])
def predict():
    img = Image.open(request.files['file']).convert('L').resize((64,64))
    img = np.array(img).reshape(1, -1)
    pred = model.predict(img)[0]
    return f"Predicted class: {pred}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
