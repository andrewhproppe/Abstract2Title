from src.model.gpt2_model import load_gpt2_model

model, tokenizer = load_gpt2_model()

input_text = "Hugging Face is creating"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=30)

print(tokenizer.decode(outputs[0]))