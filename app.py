from flask import Flask, request, render_template
from simpletransformers.seq2seq import Seq2SeqModel

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained BART model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="src/pipeline/model_training/outputs",  # Update path if necessary
)


# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the abstract submitted by the user
        abstract = request.form['abstract']

        # Error handling to ensure the input is a valid string
        if not isinstance(abstract, str) or not abstract.strip():
            error_message = "Invalid input. Please enter a valid abstract as text."
            return render_template('index.html', error=error_message)

        try:
            # Make a prediction using the model
            predicted_title = model.predict([abstract.strip()])[0]
        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"
            return render_template('index.html', error=error_message)

        return render_template('result.html', abstract=abstract, predicted_title=predicted_title)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)