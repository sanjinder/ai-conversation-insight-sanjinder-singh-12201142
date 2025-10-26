## Project Overview

AI Conversation Insight: Real-Time Chat & Customer Feedback Analytics
> AI Conversation Insight is a robust, end-to-end web application designed to instantly analyze large volumes of unstructured customer feedback—including chat transcripts, reviews, and survey responses—to extract actionable sentiment and categorize feedback for rapid business decision-making.
>
 ## Problem Statement
> Businesses often struggle to efficiently process the immense flow of customer interactions across multiple channels. Manually sifting through thousands of chat logs or feedback entries is slow, costly, and prone to human error, leading to delayed responses to critical customer pain points and missed opportunities for product improvement.
>
## Solution Summary
> We built a lightweight Python and Streamlit web application that integrates with a state-of-the-art, pre-trained Hugging Face language model (specifically, a DistilBERT-based text classification model) to perform batch and live sentiment analysis. The solution takes raw text data (either uploaded CSV/Excel files or pasted text), automatically processes it, and returns clean, structured data complete with sentiment labels, confidence scores, and high-level actionable next steps.
>
 ## Tech Stack
List all technologies, frameworks, APIs, and tools you used.
- Backend/Core Logic: Python 3.10
- Frontend/UI: Streamlit (for rapid prototyping and interactive data visualization)
- Data Handling: Pandas
- LLM / AI Models: Hugging Face Inference API (using distilbert-base-uncased-finetuned-sst-2-english for core sentiment analysis)
- External Libraries: requests for API calls, dotenv for secret management.
- Version Control: Git + GitHub

  ## Project Structure
  
root/
├── app.py                      # Main entry point and Streamlit app logic
├── dataset_analyzer.py         # Module for handling dataset uploads and batch analysis
├── navigation.py               # Module for managing multi-page Streamlit navigation
├── requirements.txt            # All Python dependencies
└── README.md                   # This file


## Setup Instructions (with Conda)

Follow these exact steps to run your project locally.
```bash
# 1. Clone the repository
git clone https://github.com/<[your-repo-link](https://github.com/sanjinder/ai-conversation-insight-sanjinder-singh-12201142)>.git
cd <repo-folder>
# 3. Create and activate a environment
# 2. Install dependencies
pip install -r requirements.txt
# 3. Run the app
streamlit run app.py

```
## Features
- End-to-end working web app (accessible via browser): A complete, interactive Streamlit application.
- Batch Sentiment Processing: Upload a CSV or Excel file and process hundreds of feedback entries with a single click.
- Actionable Insights: Generates context-specific next steps for POSITIVE (e.g., "Reinforce successful strategies") and NEGATIVE (e.g., "Investigate customer pain points") feedback.
- Real-Time Analysis: Instantly analyze single or multi-line chat transcripts directly in the "Live Chat Analyzer."
- Clear, maintainable code structure
- Exportable Results: Download the analyzed dataset, including new columns for Sentiment, Confidence Score, and Action, as a clean CSV file.

## 🧩 Technical Architecture
The application follows a simple client-server architecture facilitated by Streamlit and the Hugging Face API.
> - User (Web)
> - Streamlit Frontend
> - Python Backend/Core  
> - Hugging Face Inference (DistilBERT Model)
> -  Sentiment & Action Logic (App)
-  Frontend sends text (live or batch) to the Python core logic.
-  Python Backend uses the requests library to query the Hugging Face Inference Endpoint with the text.
-  The AI Model returns the sentiment and confidence score.
-  The Action Logic maps the sentiment (Positive/Negative) to a predefined, business-focused actionable recommendation.
-  Results are displayed and made available for download in the Frontend.

## 🧾 References
- Hugging Face Models
- Streamlit Documentation

  










