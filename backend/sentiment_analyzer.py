import pandas as pd
from transformers import pipeline
import os
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS
from typing import Union
import google.generativeai as genai

load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Globals to store processed data ---
processed_df = None
feedback_summary = None
# --- End Globals ---

# Initialize the sentiment analysis pipeline
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"Error initializing sentiment pipeline: {e}")
    sentiment_pipeline = None


def get_sentiment_score(feedback_text: str) -> int:
    """
    Returns:
    - 1 for Positive sentiment with confidence above threshold
    - -1 for Negative sentiment with confidence above threshold
    - 0 for Neutral or low confidence results
    """
    if sentiment_pipeline is None:
        print("Sentiment pipeline not initialized. Returning neutral (0).")
        return 0  # Default to Neutral

    if not feedback_text or not isinstance(feedback_text, str) or feedback_text.strip() == "":
        return 0  # Neutral for empty/invalid text

    try:
        result = sentiment_pipeline(feedback_text)[0]
        label = result['label']
        confidence = result['score']

        CONFIDENCE_THRESHOLD = 0.35  # You can tweak this threshold if needed

        if label == 'POSITIVE':
            return 1 if confidence >= CONFIDENCE_THRESHOLD else 0
        elif label == 'NEGATIVE':
            return -1 if confidence >= CONFIDENCE_THRESHOLD else 0
        else:
            # fallback neutral
            return 0
    except Exception as e:
        print(f"Error during sentiment polarity calculation: {e}")
        return 0


def add_sentiment_column_to_csv(csv_filepath: str) -> Union[pd.DataFrame, None]:
    global processed_df
    if sentiment_pipeline is None:
        print("Cannot process CSV: Sentiment pipeline failed to initialize.")
        return None
    try:
        df = pd.read_csv(csv_filepath)

        required_cols = ['Customer ID', 'Date', 'Feedback Text', 'Rating']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: '{col}' column not found in {csv_filepath}.")
                return None

        sentiment_score_column_name = 'Sentiment'  # Consistent column name

        print(f"Processing sentiment for records in {csv_filepath}...")

        df[sentiment_score_column_name] = df['Feedback Text'].fillna('').astype(str).apply(get_sentiment_score)

        columns_to_keep = required_cols + [sentiment_score_column_name]
        output_df = df[columns_to_keep]

        # Save back to CSV
        output_df.to_csv(csv_filepath, index=False)
        print(f"Successfully updated '{csv_filepath}' with columns: {', '.join(columns_to_keep)}")

        processed_df = output_df
        return output_df

    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
    except Exception as e:
        print(f"Error processing CSV {csv_filepath}: {e}")
    return None


def generate_gemini_summary(df: pd.DataFrame) -> Union[str, None]:
    global feedback_summary
    if df is None or df.empty:
        print("DataFrame is empty. No feedback to summarize.")
        return None

    sentiment_column_name = 'Sentiment'
    if sentiment_column_name not in df.columns:
        print(f"Error: Column '{sentiment_column_name}' not found in DataFrame.")
        return None

    feedback_data_for_prompt = []
    for index, row in df.iterrows():
        feedback_entry = f"Review {index + 1}:\n"
        feedback_entry += f"  Text: {row.get('Feedback Text', 'N/A')}\n"
        feedback_entry += f"  Original Rating: {row.get('Rating', 'N/A')}\n"
        feedback_entry += f"  Sentiment Score(-1,0,1): {row.get(sentiment_column_name, 'N/A')}\n"
        feedback_data_for_prompt.append(feedback_entry)

    reviews_string = "\n".join(feedback_data_for_prompt)
    if not reviews_string.strip():
        print("No valid feedback data extracted to send for summarization.")
        return None

    prompt = f"""
You are an AI assistant tasked with summarizing customer feedback.
Below is a list of customer reviews, including their feedback text, original rating, and a 'Sentiment Score(-1,0,1)' where -1 is Negative, 0 is Neutral, and +1 is Positive.
Please provide a concise summary of these customer reviews. Your summary should highlight:
1. Overall customer sentiment based on the 'Sentiment Score(-1,0,1)'.
2. Key positive themes or common praises (associated with a score of +1).
3. Key negative themes or common complaints/areas for improvement (associated with a score of -1).
4. Any notable trends or interesting points, considering neutral feedback (a score of 0) as well.
Customer Feedback Data:
{reviews_string}
Summary:
"""

    print("\nSending data to Gemini for summarization...")
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)

        summary = response.text
        feedback_summary = summary
        return summary
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return None


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)


@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    global processed_df, feedback_summary
    if processed_df is None:
        return jsonify({
            "error": "Data not processed yet. Please try again shortly or ensure the backend started correctly."
        }), 500
    if feedback_summary is None:
        return jsonify({
            "error": "Summary not generated yet. Please try again shortly or ensure the backend started correctly."
        }), 500

    data_json = processed_df.to_dict(orient='records')
    return jsonify({
        "feedbackData": data_json,
        "summary": feedback_summary
    })


def initial_data_processing():
    global processed_df, feedback_summary
    print("--- Initializing Backend: Processing Data ---")
    feedback_csv_file = r"C:\Users\gagan\Downloads\AI-Based-Customer-Feedback-Summarizer\backend\feeedback.csv"

    current_df = add_sentiment_column_to_csv(feedback_csv_file)
    if current_df is not None:
        print("\n--- Generating Initial Gemini Summary ---")
        generate_gemini_summary(current_df)
        if feedback_summary:
            print("\n--- Gemini Customer Feedback Summary (Initial) ---")
            print(feedback_summary)
            print("--------------------------------------")
        else:
            print("Failed to generate initial summary.")
    else:
        print("Skipping summary generation due to errors in sentiment analysis or CSV processing.")
    print("--- Backend Data Processing Complete ---")


if __name__ == "__main__":
    initial_data_processing()
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
