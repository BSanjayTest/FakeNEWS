from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import traceback # For detailed error logging
import re # Keep for potential future basic checks if needed
import requests # For web scraping
from bs4 import BeautifulSoup # For parsing scraped HTML
import numpy as np # Needed for similarity calculation
import pandas as pd # Needed for loading processed data
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Import VADER
from sklearn.exceptions import NotFittedError # Import error type
from sklearn.feature_extraction.text import TfidfVectorizer # Add for snippet analysis
try:
    # Import transformers and torch
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: transformers or torch not found. Please install dependencies: pip install -r backend/requirements.txt")
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None

# --- Configuration ---
MAX_SCRAPED_RESULTS = 5
SCRAPE_TIMEOUT = 5 # Seconds
WORDS_FOR_SCRAPING_QUERY = 10 # Use first 10 words for query
SIMILARITY_THRESHOLD = 0.5 # Minimum cosine similarity to be considered 'similar'
MAX_SIMILAR_RESULTS = 3 # Show top N similar fake/true
MAX_EXCESSIVE_PUNCT = 5 # Threshold for flagging punctuation
MAX_ALL_CAPS_WORDS = 5 # Threshold for flagging all caps words
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Model identifier for Hugging Face
LLM_MAX_NEW_TOKENS = 100 # Max tokens for the LLM to GENERATE
LLM_TEMPERATURE = 0.7
# LLM_CONTEXT_SIZE removed, handled by model implicitly to some extent
# --- End Configuration ---

# Define paths relative to the app.py file location
backend_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(backend_dir, '..', 'models')
data_dir = os.path.join(backend_dir, '..', 'data')
model_path = os.path.join(models_dir, 'model_pipeline.pkl')
vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
processed_data_path = os.path.join(data_dir, 'processed_news_data.pkl')
# llm_model_path removed, transformers handles caching

# --- Load Models and Data ---
vader_analyzer = SentimentIntensityAnalyzer()
classifier = None
vectorizer = None
processed_data_df = None
all_text_vectors = None
llm_model = None # Transformers model object
llm_tokenizer = None # Transformers tokenizer object

try:
    print(f"Attempting to load classifier model pipeline structure from: {model_path}")
    if os.path.exists(model_path):
        _pipeline_loaded = joblib.load(model_path)
        if len(_pipeline_loaded.steps) >= 2:
            classifier = _pipeline_loaded.steps[1][1]
            print("Classifier loaded successfully from pipeline.")
        else:
             print("Error: Loaded pipeline does not have expected structure.")
    else:
        print(f"Error: Classifier pipeline file not found at {model_path}")
except Exception as e:
    print(f"Error loading classifier pipeline: {e}")

try:
    print(f"Attempting to load vectorizer from: {vectorizer_path}")
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        print("TF-IDF vectorizer loaded successfully.")
    else:
        print(f"Error: Vectorizer file not found at {vectorizer_path}")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

try:
    print(f"Attempting to load processed data from: {processed_data_path}")
    if os.path.exists(processed_data_path):
        processed_data_df = joblib.load(processed_data_path)
        print(f"Processed data loaded successfully ({len(processed_data_df)} articles).")
        if vectorizer is not None and not processed_data_df.empty:
            print("Calculating TF-IDF vectors for similarity search...")
            all_text_vectors = vectorizer.transform(processed_data_df['text'])
            print("TF-IDF vectors calculated.")
        else:
            print("Skipping TF-IDF vector calculation (vectorizer or data missing).")
    else:
        print(f"Error: Processed data file not found at {processed_data_path}")
except Exception as e:
    print(f"Error loading processed data: {e}")

# --- Load LLM using Transformers (Experimental) ---
if AutoModelForCausalLM is not None and AutoTokenizer is not None and torch is not None:
    print(f"Attempting to load LLM model and tokenizer: {LLM_MODEL_NAME}")
    print("This may take time and download model data on the first run.")
    print("Attempting to use GPU (CUDA) if available.")
    try:
        # Check available RAM (very rough estimate)
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            print(f"Estimated available RAM before loading LLM: {available_ram_gb:.2f} GB")
            # Adjust RAM warning if targeting GPU VRAM primarily
            # if available_ram_gb < 5:
            #     print("WARNING: Available RAM seems low, LLM loading might fail or cause issues.")
        except ImportError:
            print("psutil not found, cannot check RAM. pip install psutil")
        except Exception as ram_e:
            print(f"Could not estimate RAM: {ram_e}")

        # Determine device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            # Use float16 on GPU for better performance and lower VRAM, if supported
            # Some older GPUs might require float32
            try:
                 # Check if float16 is supported
                 _ = torch.tensor([1.0], dtype=torch.float16).to(device)
                 dtype = torch.float16
                 print("CUDA available. Using GPU with float16 precision.")
            except Exception as e:
                 print(f"CUDA available, but float16 may not be fully supported ({e}). Falling back to float32.")
                 dtype = torch.float32
        else:
            device = "cpu"
            dtype = torch.float32 # Always use float32 on CPU
            print("CUDA not available. Using CPU with float32 precision. (Expect slow performance)")

        # Load tokenizer (usually doesn't need specific device/dtype)
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

        # Load model using device_map="auto" with accelerate library
        # This lets accelerate handle placing layers on CPU/GPU optimally
        # Specify dtype for potential VRAM savings on GPU
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" # Requires `accelerate` library
        )
        # No need for explicit .to(device) when using device_map="auto"

        llm_model.eval()
        # Verify the final device placement (might be mixed CPU/GPU)
        print(f"LLM model loaded. Final device map: {llm_model.hf_device_map}")

    except MemoryError as mem_err:
         # Check if it's CUDA OOM
         if "cuda out of memory" in str(mem_err).lower():
             print("CRITICAL ERROR: CUDA Out of Memory (GPU VRAM likely insufficient).")
             print("Try a smaller model quantization or ensure GPU has enough VRAM (~2GB+ for float16, ~4GB+ for float32 for TinyLlama).")
         else:
             print("CRITICAL ERROR: System Out of Memory while loading LLM.")
         print("LLM functionality disabled.")
         llm_model = None
         llm_tokenizer = None
    except Exception as e:
        print(f"CRITICAL ERROR loading LLM with transformers: {e}")
        print(traceback.format_exc())
        llm_model = None
        llm_tokenizer = None
else:
    print("LLM loading skipped because transformers or torch could not be imported.")
# --- End Loading ---

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# --- Analysis Helper Functions ---
def analyze_sentiment(text):
    try:
        vs = vader_analyzer.polarity_scores(text)
        # Determine overall sentiment
        compound = vs['compound']
        if compound >= 0.05:
            sentiment_label = "Positive"
        elif compound <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        return {"label": sentiment_label, "score": compound, "details": vs}
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return {"label": "Error", "score": 0, "details": {}}

def find_similar_articles(input_text):
    if vectorizer is None or all_text_vectors is None or processed_data_df is None or processed_data_df.empty:
        return {"similar_true": [], "similar_fake": []}
    try:
        input_vector = vectorizer.transform([input_text])
        similarities = cosine_similarity(input_vector, all_text_vectors)[0]
        processed_data_df['similarity'] = similarities
        similar_df = processed_data_df[processed_data_df['similarity'] >= SIMILARITY_THRESHOLD].copy()
        top_true = similar_df[similar_df['label'] == 1].nlargest(MAX_SIMILAR_RESULTS, 'similarity')
        top_fake = similar_df[similar_df['label'] == 0].nlargest(MAX_SIMILAR_RESULTS, 'similarity')
        processed_data_df.drop(columns=['similarity'], inplace=True)

        format_result = lambda row: f"{row['title']} (Similarity: {row['similarity']:.2f}) - Snippet: {row['text'][:100]}..."

        # FIX: Explicitly check if DataFrames are empty before applying
        similar_true_list = []
        if not top_true.empty:
            try:
                similar_true_list = list(top_true.apply(format_result, axis=1))
            except Exception as apply_err:
                print(f"Error applying format to top_true: {apply_err}")
                similar_true_list = ["Error formatting similar true articles."]

        similar_fake_list = []
        if not top_fake.empty:
            try:
                similar_fake_list = list(top_fake.apply(format_result, axis=1))
            except Exception as apply_err:
                print(f"Error applying format to top_fake: {apply_err}")
                similar_fake_list = ["Error formatting similar fake articles."]

        return {"similar_true": similar_true_list, "similar_fake": similar_fake_list}
    except Exception as e:
        print(f"Error during similarity search: {e}")
        print(traceback.format_exc())
        if 'similarity' in processed_data_df.columns:
            processed_data_df.drop(columns=['similarity'], inplace=True)
        return {"similar_true": ["Error during similarity search."], "similar_fake": ["Error during similarity search."]}

def check_basic_linguistics(text):
    excessive_punct = len(re.findall(r'[!\?]{2,}', text)) # Find !! ?? !? ?! etc.
    all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text)) # Find words with 2+ uppercase letters
    flags = []
    if excessive_punct >= MAX_EXCESSIVE_PUNCT:
        flags.append(f"High amount of excessive punctuation found ({excessive_punct}).")
    if all_caps_words >= MAX_ALL_CAPS_WORDS:
        flags.append(f"High amount of all-caps words found ({all_caps_words}).")

    return flags if flags else ["No basic linguistic flags detected."]

# --- Web Scraping Function ---
def scrape_duckduckgo(query):
    if not query:
        return {"status": "error", "message": "No query provided for scraping."}

    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    results = []
    print(f"Scraping DuckDuckGo for: {query}")
    try:
        response = requests.get(search_url, headers=headers, timeout=SCRAPE_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find result links (selectors might change if DDG updates HTML version)
        links = soup.select('div.result__body h2.result__title a.result__a')
        snippets = soup.select('div.result__body a.result__snippet')

        for i in range(min(len(links), len(snippets), MAX_SCRAPED_RESULTS)):
            title = links[i].get_text(strip=True)
            link = links[i]['href']
            snippet = snippets[i].get_text(strip=True)
            results.append({'title': title, 'link': link, 'snippet': snippet})

        if not results:
             return {"status": "success", "message": "No results found on DuckDuckGo HTML.", "data": []}

        print(f"Scraping successful, found {len(results)} results.")
        return {"status": "success", "message": f"{len(results)} results found.", "data": results}

    except requests.exceptions.Timeout:
        print(f"Scraping timed out for query: {query}")
        return {"status": "error", "message": "Search engine query timed out."}
    except requests.exceptions.RequestException as e:
        print(f"Scraping failed for query '{query}': {e}")
        return {"status": "error", "message": f"Could not connect to search engine: {e}"}
    except Exception as e:
        print(f"Error parsing scrape results for query '{query}': {e}")
        print(traceback.format_exc())
        return {"status": "error", "message": "Failed to parse search results."}

# --- LLM Analysis Function (Using Transformers) ---
def get_llm_analysis(text, scraped_snippets):
    if llm_model is None or llm_tokenizer is None or torch is None:
        return "LLM analysis disabled (model/tokenizer not loaded)."

    print("Starting LLM analysis (Transformers)...")
    start_time = pd.Timestamp.now()

    # Format prompt for TinyLlama Chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant analyzing news credibility."},
        {"role": "user", "content": f"Analyze the following news text. Is it likely true or fake? Consider the provided search result snippets if available. Provide a brief justification.\n\nNews Text: '{text}'\n"}
    ]
    if scraped_snippets:
         messages[1]["content"] += "\nRelevant Search Snippets:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(scraped_snippets[:2])])

    # Prepare prompt using tokenizer's chat template
    prompt_text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        device = llm_model.device # Get device model is on
        inputs = llm_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(device) # Truncate long inputs

        # Generate text
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=True, # Use sampling for temperature > 0
                pad_token_id=llm_tokenizer.eos_token_id # Avoid padding warning
            )
        # Decode only the newly generated tokens
        output_text = llm_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        analysis = output_text.strip()
        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()
        print(f"LLM analysis complete in {duration:.2f} seconds.")
        return analysis if analysis else "LLM returned empty analysis."

    except RuntimeError as OOM_error:
        if "out of memory" in str(OOM_error).lower():
             print("ERROR: CUDA Out of Memory during LLM inference (if using GPU).")
             return "Error: Out of memory during LLM analysis."
        else:
             print(f"Runtime Error during LLM inference: {OOM_error}")
             print(traceback.format_exc())
             return "Runtime Error during LLM analysis."
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        print(traceback.format_exc())
        return "Error during LLM analysis."

# --- Flask Routes ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Missing "text" in request body'}), 400

    text_to_analyze = request.json['text']

    # Initialize response dictionary with new fields
    response_data = {
        'analyzed_text': text_to_analyze,
        'classification': None,
        'classification_confidence': None,
        'sentiment': {},
        'basic_linguistic_flags': [],
        'similarity_results': {},
        'scraped_results': {'status': 'not_attempted', 'message': 'N/A', 'data': []},
        'scraped_analysis': {'keywords': [], 'sources': []}, # New field for snippet analysis
        'llm_analysis': "Not performed." # Add new field
    }

    # --- 1. Classifier Prediction ---
    if classifier is None or vectorizer is None:
        print("Prediction attempted but classifier or vectorizer is not loaded.")
        response_data['classification'] = "Error: Classifier or Vectorizer not loaded."
    else:
        try:
            # Transform input text using the loaded vectorizer
            vectorized_input = vectorizer.transform([text_to_analyze])
            # Predict using the loaded classifier
            prediction_label = classifier.predict(vectorized_input)[0]
            prediction_proba = classifier.predict_proba(vectorized_input)[0]

            if prediction_label == 1:
                response_data['classification'] = "Likely True (based on dataset)"
                response_data['classification_confidence'] = prediction_proba[1]
            else:
                response_data['classification'] = "Likely Fake (based on dataset)"
                response_data['classification_confidence'] = prediction_proba[0]
        except NotFittedError as nfe:
            print(f"Error during classification - Model/Vectorizer might not be fitted correctly: {nfe}")
            print(traceback.format_exc())
            response_data['classification'] = "Error: Model reported as not fitted."
        except Exception as e:
            print(f"Error during classification: {e}")
            print(traceback.format_exc())
            response_data['classification'] = "Error during classification."

    # --- 2. Sentiment Analysis ---
    response_data['sentiment'] = analyze_sentiment(text_to_analyze)

    # --- 3. Basic Linguistic Checks ---
    response_data['basic_linguistic_flags'] = check_basic_linguistics(text_to_analyze)

    # --- 4. Similarity Search ---
    response_data['similarity_results'] = find_similar_articles(text_to_analyze)

    # --- 5. Web Scraping (using first N words) ---
    if not text_to_analyze.strip():
        print("Skipping scraping: Input text is empty.")
        response_data['scraped_results'] = {'status': 'skipped', 'message': 'Input text was empty.', 'data': []}
    else:
        # Extract first N words for the search query
        words = text_to_analyze.split()
        search_query = " ".join(words[:WORDS_FOR_SCRAPING_QUERY])

        if not search_query:
             print("Skipping scraping: Could not generate search query from text.")
             response_data['scraped_results'] = {'status': 'skipped', 'message': 'Could not generate search query.', 'data': []}
        else:
            print(f"Attempting to scrape based on first {WORDS_FOR_SCRAPING_QUERY} words: '{search_query}'")
            scrape_result = scrape_duckduckgo(search_query)
            response_data['scraped_results'] = scrape_result

    # --- 5b. Analyze Scraped Snippets --- (New)
    scraped_sources = [] # Store full source info
    if scrape_result['status'] == 'success' and scrape_result['data']:
        scraped_sources = scrape_result['data'] # Keep title, link, snippet

    # --- 5c. Analyze Scraped Snippets --- (New)
    if scraped_sources:
        try:
            snippets = [item['snippet'] for item in scraped_sources]
            if len(snippets) > 1: # Need at least 2 snippets for TF-IDF to be meaningful
                snippet_vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
                tfidf_matrix = snippet_vectorizer.fit_transform(snippets)
                feature_names = snippet_vectorizer.get_feature_names_out()
                # Sum TF-IDF scores across all snippets for each term
                total_scores = tfidf_matrix.sum(axis=0).A1 # .A1 converts matrix row to flat numpy array
                # Get indices of top N scores
                num_keywords = min(10, len(feature_names))
                top_indices = np.argsort(total_scores)[-num_keywords:][::-1]
                top_keywords = [feature_names[i] for i in top_indices]
                response_data['scraped_analysis']['keywords'] = top_keywords
            else:
                response_data['scraped_analysis']['keywords'] = ["Not enough context from snippets for keyword analysis."]
        except Exception as snippet_e:
            print(f"Error analyzing scraped snippets: {snippet_e}")
            response_data['scraped_analysis']['keywords'] = ["Error during snippet analysis."]
        response_data['scraped_analysis']['sources'] = scraped_sources # Pass sources along
    else:
        response_data['scraped_analysis']['keywords'] = ["No search results to analyze."]
        response_data['scraped_analysis']['sources'] = []

    # --- 6. LLM Analysis (Using Transformers) ---
    llm_input_snippets = [item['snippet'] for item in scraped_sources] # Use snippets from stored sources
    response_data['llm_analysis'] = get_llm_analysis(text_to_analyze, llm_input_snippets)

    return jsonify(response_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    if classifier is None or vectorizer is None or processed_data_df is None or all_text_vectors is None:
         print("\nWARNING: Server starting but one or more required models/data assets failed to load.")
         print("Classification and/or similarity search WILL NOT WORK. Check logs above.")
    else:
        print("\nAll required models and data loaded successfully.")

    # Check for LLM via Transformers
    if AutoModelForCausalLM is not None and (llm_model is None or llm_tokenizer is None):
        print("\nWARNING: LLM (Transformers) was not loaded successfully (check errors above). LLM analysis will be disabled.")
    elif AutoModelForCausalLM is None:
        print("\nINFO: transformers/torch not installed/imported. LLM analysis is disabled.")
    else:
        print(f"\nLLM (Transformers) loaded and ready for analysis (Device: {llm_model.device if llm_model else 'None'}).")

    print(f"\nStarting Flask server on port {port} with debug mode: {debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 