import os
import streamlit as st
import requests
from dotenv import load_dotenv
import time # Import time for potential retries (optional but good practice)

# --- CRITICAL FIX: Global variables are now defined inside main() ---
# This ensures environment variables are loaded only when the function runs,
# preventing crashes during initial module import.

def get_api_config():
    """Loads env vars and returns API configuration, raising an error if the key is missing."""
    # load_dotenv() is called here to ensure environment variables are available when this function runs
    load_dotenv()
    # The variable must match the name used in your .env file
    hf_api_key = os.getenv("HF_API_KEY")

    if not hf_api_key:
        raise ValueError("HF_API_KEY is missing. Please set it in your .env file.")

    # --- MODEL FIX: Using a stable, public RoBERTa model for sentiment analysis ---
    api_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    return api_url, headers

def query_huggingface(texts, api_url, headers, retries=2, delay=5):
    """
    Queries the Hugging Face API for a list of texts with basic retry logic.
    """
    last_exception = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": texts}, timeout=20) # Added timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            # API returns a list of prediction lists (e.g., [[{pred1}], [{pred2}]])
            return response.json()
        except requests.exceptions.HTTPError as e:
            last_exception = e
            # Specifically handle 404 - Model Not Found or Endpoint Issue
            if e.response.status_code == 404:
                st.error(f"HTTP Error 404: Model endpoint not found at {api_url}. Please check the model name or Hugging Face status.")
                # No point retrying a 404
                raise e
            # Handle potential rate limits or temporary server issues (like 503)
            elif e.response.status_code in [429, 503]:
                st.warning(f"API request failed (Status {e.response.status_code}). Retrying in {delay}s... (Attempt {attempt + 1}/{retries + 1})")
                time.sleep(delay)
            else:
                # Other HTTP errors
                raise e # Re-raise other HTTP errors immediately
        except requests.exceptions.RequestException as e:
            # Catch other connection/timeout errors
            last_exception = e
            st.warning(f"Connection error: {e}. Retrying in {delay}s... (Attempt {attempt + 1}/{retries + 1})")
            time.sleep(delay)

    # If all retries fail, raise the last encountered exception
    if last_exception:
        raise last_exception
    return [] # Should not be reached, but ensures return type


def generate_action(label):
    """
    Provides a human-readable action plan based on the sentiment label.
    Note: LABEL_0 is Negative, LABEL_1 is Neutral, LABEL_2 is Positive for this model.
    """
    if label in ["LABEL_2", "POSITIVE"]:
        return "✅ Reinforce successful strategies. (Positive Feedback)"
    elif label in ["LABEL_0", "NEGATIVE"]:
        return "⚠️ Investigate and resolve customer pain points. (Negative Feedback)"
    else: # LABEL_1, NEUTRAL, or unexpected
        return "🔍 Monitor feedback for more data points. (Neutral/Mixed Feedback)"

# --- DEFINITION OF main() FUNCTION ---
def main():
    st.title("📣 Live Chat Analyzer")
    st.markdown("Enter customer feedback or chat transcripts below. Each line will be analyzed separately using **Hugging Face's RoBERTa Sentiment** model.")

    API_URL = "" # Initialize to avoid NameError if config fails
    HEADERS = {}

    try:
        # Load config dynamically inside main()
        API_URL, HEADERS = get_api_config()
    except ValueError as e:
        st.error(f"🚨 CONFIGURATION ERROR: {e}")
        st.info("Please verify your `.env` file is in the root directory and the `HF_API_KEY` is set correctly.")
        return # Halt execution if key is missing
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred during setup: {e}")
        return


    feedback_input = st.text_area(
        "Customer Feedback",
        height=200,
        placeholder="E.g., \nI love the new design!\n\nThe service was terrible, I waited 30 minutes."
    )

    if st.button("Analyze Feedback"):
        # Remove empty lines and leading/trailing whitespace
        lines = [line.strip() for line in feedback_input.splitlines() if line.strip()]

        if not lines:
            st.warning("Please enter at least one line of feedback.")
            return

        with st.spinner('Contacting Hugging Face API and analyzing feedback...'):
            try:
                # Pass API_URL and HEADERS loaded within main()
                results = query_huggingface(lines, API_URL, HEADERS)

            except requests.exceptions.HTTPError as e:
                # More specific error message based on status code
                if e.response.status_code == 401:
                    st.error(f"HTTP Error 401: Unauthorized. Your Hugging Face API key is likely invalid or missing permissions.")
                elif e.response.status_code == 404:
                    st.error(f"HTTP Error 404: The requested model endpoint was not found. Please verify the model name.")
                elif e.response.status_code == 429:
                    st.error(f"HTTP Error 429: Too many requests. You might be hitting rate limits.")
                elif e.response.status_code == 503:
                    st.error(f"HTTP Error 503: Service Unavailable. The Hugging Face model might be loading or temporarily down. Please try again later.")
                else:
                    st.error(f"HTTP Error: {e}. Status Code: {e.response.status_code}")
                return # Stop processing on HTTP error
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: Could not connect to the Hugging Face API. Details: {e}")
                return
            except Exception as e:
                st.error(f"An unexpected error occurred during API call: {e}")
                return

            st.subheader("Analysis Results")

            # Process the results (API returns a list of lists of dictionaries)
            processed_results = []
            if isinstance(results, list):
                for sublist in results:
                    if isinstance(sublist, list) and sublist:
                        # Take the first prediction (highest score)
                        processed_results.append(sublist[0])
                    elif isinstance(sublist, dict):
                         # Handle case where API might return a flat list of dicts (less common)
                        processed_results.append(sublist)


            if len(lines) != len(processed_results):
                st.warning(f"Mismatch between input lines ({len(lines)}) and API results ({len(processed_results)}). Some lines might not have been analyzed correctly.")
                # Attempt to pair based on available results
                results_to_display = zip(lines[:len(processed_results)], processed_results)
            else:
                results_to_display = zip(lines, processed_results)


            for i, (text, best_prediction) in enumerate(results_to_display, start=1):
                if isinstance(best_prediction, dict) and 'label' in best_prediction and 'score' in best_prediction:
                    # Map the model's raw labels (LABEL_0, LABEL_1, LABEL_2) to readable ones
                    readable_label = best_prediction['label'].replace("LABEL_0", "NEGATIVE").replace("LABEL_1", "NEUTRAL").replace("LABEL_2", "POSITIVE")
                    
                    st.markdown(f"**{i}. Feedback:** `{text}`")
                    st.write(f"**Sentiment:** **{readable_label}** | **Confidence:** `{best_prediction['score']:.4f}`")
                    st.info(f"**Action Plan:** {generate_action(best_prediction['label'])}")
                    st.markdown("---")
                else:
                    st.warning(f"Could not properly analyze: '{text}' - Unexpected result structure: {best_prediction}")
                    st.markdown("---")


# --- ENSURE THIS LINE IS PRESENT ---
if __name__ == "__main__":
    main()
