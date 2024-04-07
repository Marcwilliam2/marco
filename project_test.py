from flask import Flask, request, jsonify
from fastai.vision.all import *
from io import BytesIO
from PIL import Image
from pyngrok import ngrok  # Import Ngrok

ngrok.set_auth_token("2emw600hVIiCaCnjg82wtdaNI7U_78ASLSYN62q19t8nqAVz9")

# Load the FastAI learner
learn = load_learner("fastai_model2.pkl")

# Function to predict label for an image
def predict_image(img):
    predicted_label, _, _ = learn.predict(img)
    return predicted_label

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        img = PILImage.create(BytesIO(file.read()))
        predicted_label = predict_image(img)

        return jsonify({'predicted_label': str(predicted_label)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    ngrok_tunnel = ngrok.connect(5000)
    print('Public URL:', ngrok_tunnel.public_url)  
    app.run()
