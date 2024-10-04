import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from src.data.data import fetch_papers_for_training

# Check if a GPU is available
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Load the fine-tuned model and tokenizer
model = BartForConditionalGeneration.from_pretrained('../model_training/fine_tuned_bart_model')
tokenizer = BartTokenizer.from_pretrained('../model_training/fine_tuned_bart_tokenizer')

# Move model to the correct device
model = model.to(device)
model.eval()  # Set the model to evaluation mode


def generate_title(abstract, model, tokenizer, max_length=30):
    # Tokenize the abstract
    inputs = tokenizer(
        abstract, return_tensors='pt', truncation=True, max_length=512, padding='max_length'
    ).to(device)

    # Generate a title with adjusted parameters
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,  # Limit the max length for titles
        num_beams=5,  # Beam search to explore better title options
        no_repeat_ngram_size=2,  # Avoid repeated n-grams
        early_stopping=True,  # Stop when the model predicts the end of the sequence
    )

    # Decode the generated title properly
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return title


# Example abstract to test
abstract = "This paper proposes a novel approach to handling missing data in machine learning models by introducing an adaptive imputation method."

# Generate the title for the given abstract
generated_title = generate_title(abstract, model, tokenizer)
print(f"Generated Title: {generated_title}")