from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Load the trained model and tokenizer
model_path = "distilled_bert_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to preprocess text input
def preprocess_text(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)

# Function to make predictions
def predict(text):
    model.eval()  # Set model to evaluation mode
    inputs = preprocess_text(text)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    prediction = torch.argmax(outputs.logits, dim=-1).item()  # Get the predicted label
    return prediction  # Returns a label index (e.g., 0 = not toxic, 1 = toxic)

# Route to serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# API route for text classification
@app.route("/predict", methods=["POST"])
def classify_text():
    text = request.form.get("text")  # Get input text from the form
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    prediction = predict(text)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
