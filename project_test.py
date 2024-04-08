from flask import Flask, request, jsonify
from fastai.vision.all import *
from io import BytesIO
from PIL import Image
from pyngrok import ngrok  # Import Ngrok
import os
from azure.storage.blob import BlobServiceClient

# Load the FastAI learner
# learn = load_learner("fastai_model2.pkl")

connect_str = "DefaultEndpointsProtocol=https;AccountName=mystorageproject13;AccountKey=73tbI58kizlW2WzbBmWnj3m41WhUfyN4k4Nq15+tsDMVm9xTDgKtBs0eW2E+4WQ0iCXo+EeURCzX+AStY5MJ6A==;EndpointSuffix=core.windows.net"  # Connection string

container_name = "mycontainerblob"
blob_name = "fastai_model2.pkl"

# Azure Blob Service client
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Function to load model from Azure Blob Storage
def load_model_from_azure_blob():
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open("temp_model.pkl", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    try:
        return load_learner("temp_model.pkl")
    except Exception as e:
        return jsonify({"error": "Failed to load model: {}".format(str(e))}), 500

# Load the model (ensure model is in the container before starting the app)
learn = load_model_from_azure_blob()  # Call the function to load from Azure Blob

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
