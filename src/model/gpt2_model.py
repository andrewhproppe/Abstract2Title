from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_gpt2_model(model_name='gpt2'):
    # Load pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add special tokens for GPT-2 (e.g., <|endoftext|>)
special_token = tokenizer.eos_token

# Tokenize the data
def tokenize_pair(abstract, title):
    # Input is abstract, label is title
    input_ids = tokenizer.encode(abstract, return_tensors='pt')
    label_ids = tokenizer.encode(title + special_token, return_tensors='pt')  # Add end token to title
    return input_ids, label_ids