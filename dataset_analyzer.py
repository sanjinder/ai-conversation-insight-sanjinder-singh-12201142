import os
import streamlit as st
import pandas as pd
import pickle
import io
import requests
import time
from typing import List, Dict, Any
from dotenv import load_dotenv 

# --- API CONFIGURATION ---
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# New, stable API endpoint for Zero-Shot Classification
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
DEFAULT_BATCH_SIZE = 32

# --- API BATCH QUERY (Zero-Shot Classification) ---
def query_zero_shot_batch(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE, retries: int = 3, delay: int = 5) -> List[Dict[str, Any]]:
    """
    Query Hugging Face Zero-Shot Classification API in batches.
    
    This function sends texts and asks the model to classify them into the
    labels: 'positive', 'negative', or 'neutral'.
    """
    if not HF_API_KEY:
        st.error("HF_API_KEY is missing. Cannot proceed with analysis.")
        return []

    results: List[Dict[str, Any]] = []
    total_texts = len(texts)
    
    # Check if there are texts to process
    if total_texts == 0:
        return []
        
    progress_bar = st.progress(0)
    
    # Define the classification candidates once
    candidate_labels = ["positive", "negative", "neutral"]

    for i in range(0, total_texts, batch_size):
        chunk = texts[i:i+batch_size]
        
        # Payload structure for Zero-Shot classification
        payload = {
            "inputs": chunk,
            "parameters": {"candidate_labels": candidate_labels}
        }

        for attempt in range(retries + 1):
            try:
                # Increased timeout to 120 seconds for better stability on large batches
                response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120) 
                response.raise_for_status()
                chunk_results = response.json()
                
                # CRITICAL FIX: Check for the batching error (fewer results than inputs)
                if not isinstance(chunk_results, list) or len(chunk_results) != len(chunk):
                     st.error(f"Batch Error: API returned {len(chunk_results) if isinstance(chunk_results, list) else 0} results for {len(chunk)} inputs. Skipping this batch.")
                     break # Stop retrying, go to next batch
                
                # Process results: Find the highest score for the primary labels
                for result in chunk_results:
                    labels = result['labels']
                    scores = result['scores']
                    
                    # Find the index of the highest scoring label (positive, negative, or neutral)
                    best_index = scores.index(max(scores))
                    best_label = labels[best_index].upper()
                    best_score = scores[best_index]
                    
                    results.append({"label": best_label, "score": best_score})

                break # Success, exit retry loop

            except requests.exceptions.HTTPError as e:
                # Handle 401/404/429/503 errors
                status_code = e.response.status_code
                if status_code == 401:
                    st.error("Error 401: Unauthorized. Check your HF_API_KEY.")
                    return results # Return partial results on fatal auth error
                elif status_code == 404:
                    st.error("Error 404: Model not found. Check the API URL.")
                    return results
                
                # Exponential backoff for retries on temporary errors (429, 503)
                next_delay = delay * (2 ** attempt) 
                st.warning(f"⚠️ Batch starting at entry {i} failed (Status {status_code}). Retrying in {next_delay}s... (Attempt {attempt + 1}/{retries + 1})")
                if attempt < retries:
                    time.sleep(next_delay)
                else:
                    st.error(f"❌ Batch starting at entry {i} failed after all retries. Last error: {e}. Skipping this batch.")
            
            except Exception as e:
                # Catch other errors like connection issues
                next_delay = delay * (2 ** attempt) 
                st.warning(f"⚠️ Batch starting at entry {i} failed (Connection Error: {e}). Retrying in {next_delay}s... (Attempt {attempt + 1}/{retries + 1})")
                if attempt < retries:
                    time.sleep(next_delay)
                else:
                    st.error(f"❌ Batch starting at entry {i} failed after all retries. Last error: {e}. Skipping this batch.")
        
        progress_bar.progress(min((i + batch_size) / total_texts, 1.0))

    progress_bar.empty()
    return results

# --- Action Plan Generator ---
def generate_action(label: str) -> str:
    """Provides a human-readable action based on the sentiment label."""
    if label.upper() == "POSITIVE":
        return "✅ Reinforce successful strategies."
    elif label.upper() == "NEGATIVE":
        return "⚠️ Investigate and resolve customer pain points."
    else:
        return "🔍 Monitor feedback for more data points."

# --- Suggestion Generator ---
def generate_summary_suggestions(sentiments: List[str]):
    """Generates and displays strategic suggestions based on sentiment counts."""
    total = len(sentiments)
    if total == 0:
        return

    # Ensure counting is case-insensitive
    pos = sum(1 for s in sentiments if s.upper() == "POSITIVE")
    neg = sum(1 for s in sentiments if s.upper() == "NEGATIVE")

    st.markdown("### 📌 Strategic Suggestions")
    st.write(f"Out of **{total}** entries analyzed:")
    
    # Avoid division by zero if total is 0 (handled above, but good to be safe)
    if total > 0:
        st.write(f"- Positive: **{pos}** ({pos/total:.1%} of total)")
        st.write(f"- Negative: **{neg}** ({neg/total:.1%} of total)")

        if neg > pos:
            st.warning("High negative sentiment detected. The ratio is {:.1%}. Consider launching a customer recovery initiative.".format(neg/total))
        elif pos > neg:
            st.success("Positive sentiment dominates. The ratio is {:.1%}. Consider amplifying successful strategies.".format(pos/total))
        else:
            st.info("Balanced sentiment. Monitor trends and gather more feedback.")

# --- Main App ---
def main():
    st.set_page_config(page_title="Elevate Dataset Analyzer (Free API)", layout="wide")
    st.title("📊 Elevate — Dataset Sentiment Analyzer (Zero-Shot API)")
    st.markdown("Upload a `.pkl` file containing customer feedback. Analysis uses a robust, free **Zero-Shot Classification API**.")

    if not HF_API_KEY:
        st.error("🚨 Missing API Key. Please ensure your `HF_API_KEY` is set in your environment or `.env` file.")
        return
    
    with st.expander("📁 Upload Dataset", expanded=True):
        uploaded_file = st.file_uploader("Upload a `.pkl` file", type="pkl")
        
        if uploaded_file:
            try:
                # Load the DataFrame from the uploaded pickle file
                df = pickle.load(io.BytesIO(uploaded_file.getvalue()))
                
                if not isinstance(df, pd.DataFrame):
                    st.error("Uploaded file is not a valid pandas DataFrame.")
                    return
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return

            st.success(f"Loaded DataFrame with shape {df.shape}")
            st.dataframe(df.head())

            text_column = st.selectbox("Select the column containing feedback", df.columns)
            
            # Robust data cleaning: convert to string, strip whitespace, and drop empty/null values
            df_to_analyze = df[text_column].astype(str).str.strip()
            # Filter out entries that are now empty strings or NaN/None
            valid_entries = df_to_analyze[df_to_analyze.str.len() > 0]
            
            texts = valid_entries.tolist()
            indices = valid_entries.index # The indices in the original DataFrame that we are analyzing

            st.info(f"Selected column **{text_column}**. Ready to analyze **{len(texts)}** valid entries.")

            if st.button("🚀 Run Sentiment Analysis (API Call)"):
                if not texts:
                    st.warning("The selected column has no valid text entries to analyze.")
                    return
                    
                with st.spinner(f"Analyzing {len(texts)} entries using Zero-Shot Classification..."):
                    results = query_zero_shot_batch(texts) 
                
                if not results:
                    st.error("Analysis failed or returned no results. Check the console for API errors.")
                    return
                
                # Check for result length mismatch (should match the length of `texts`)
                if len(results) != len(texts):
                    st.warning(f"The analysis completed, but the number of results ({len(results)}) does not match the number of inputs ({len(texts)}). Displaying partial results.")
                    # Truncate indices and texts to match the number of successful results
                    indices = indices[:len(results)]
                    texts = texts[:len(results)]

                # Process API results
                sentiments = [r['label'] for r in results]
                confidences = [r['score'] for r in results]
                actions = [generate_action(r['label']) for r in results]

                # Update DataFrame with results using the original indices
                # We need to create Series that align with the original DataFrame index
                
                # Initialize new columns with null/NA values first
                df['Sentiment'] = pd.NA
                df['Confidence'] = pd.NA
                df['Action Plan'] = pd.NA
                
                # Set the types explicitly for robustness
                df['Confidence'] = df['Confidence'].astype('Float64') 
                
                # Assign the results back to the original DataFrame using the correct indices
                df.loc[indices, 'Sentiment'] = sentiments
                df.loc[indices, 'Confidence'] = confidences
                df.loc[indices, 'Action Plan'] = actions

                st.success("✅ Analysis complete.")
                st.dataframe(df.head(30))

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results", csv, "analyzed_feedback.csv", "text/csv")

                generate_summary_suggestions(sentiments)

if __name__ == "__main__":
    main()
