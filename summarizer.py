from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

# Load the lightweight T5-small model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_length=150, min_length=30):
    # Add prefix as required by T5 for summarization
    input_text = "summarize: " + text.strip().replace("\n", " ")

    # Tokenize input and generate summary
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
