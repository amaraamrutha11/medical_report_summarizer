from transformers import pipeline

# Use a lighter summarization model
summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarizes the input text using a DistilBART model.
    """
    if not text.strip():
        return "No input text provided."

    try:
        summary = summarizer_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {str(e)}"
