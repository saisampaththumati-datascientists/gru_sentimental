from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re 
# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('sample_model.h5')
print("Model loaded successfully!")
# Define the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON input data
    data = request.get_json(force=True)
    
    # Assume the input text is provided in the 'text' key
    input_text = data.get('text')
    input_text = input_text.lower()
    input_text = re.sub(r'[^\w\s]','',input_text)
    input_text = re.sub(r'[^a-zA-Z]', ' ', input_text)

    if not input_text:
        return jsonify({'error': 'No text provided for prediction'}), 400
    
    tokenizer = Tokenizer(num_words=5000,lower=True)
    tokenizer.fit_on_texts(input_text)
    sequences = tokenizer.texts_to_sequences(input_text)
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)


    # Use the model to predict the class of the text
    prediction = model.predict(padded_sequences)

    # Process the prediction (depending on your model's output)
    # For example, let's assume it's a binary classification:
    predicted_class = np.argmax(prediction, axis=-1)  # Get the class with the highest probability
    if predicted_class[0] == 0:
        predicted_class = 'Hair Care'
    else:
        
        predicted_class = 'Skin Care'

    return jsonify({'prediction': str(predicted_class)})
    # Return the result as a JSON response
    # return jsonify({'prediction': str(predicted_class[0])})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)