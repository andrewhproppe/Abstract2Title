import random

from transformers import TrainerCallback


class PredictionCallback(TrainerCallback):
    def __init__(self, model, tokenizer, dataset, log_interval=10, prepend_text=None):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.log_interval = log_interval
        self.prepend_text = prepend_text  # Optional text to prepend to inputs

    def on_step_end(self, args, state, control, **kwargs):
        # Generate predictions only every `log_interval` steps
        if state.global_step % self.log_interval == 0:
            # Randomly select a sample from the dataset
            sample_index = random.randint(0, len(self.dataset) - 1)
            sample_input = self.dataset[sample_index]["abstract"]
            original_title = self.dataset[sample_index]["title"]

            # Prepend text if specified (e.g., "summarize: ")
            if self.prepend_text:
                sample_input = f"{self.prepend_text} {sample_input}"

            # Tokenize the sample abstract
            inputs = self.tokenizer(
                sample_input, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            input_ids = inputs["input_ids"].to(args.device)

            # Generate a title prediction
            outputs = self.model.generate(
                input_ids, max_length=128, num_beams=4, early_stopping=True
            )

            # Decode the generated title and print it
            predicted_title = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n[Step {state.global_step}] Predicted Title: {predicted_title}")
            print(f"Original Title: {original_title}\n")
