# Fake News Detector

A simple web application to classify news text as potentially fake or true using a machine learning model trained on a custom dataset.

## Structure

- `/backend`: Contains the Flask API server and ML model loading/prediction logic.
  - `app.py`: Main Flask application.
  - `requirements.txt`: Python dependencies.
- `/frontend`: Contains the static web files (HTML, CSS, JS).
- `/data`: Place your training data CSV file here.
- `/models`: Trained ML models will be saved here.
- `train_model.py`: (To be created) Script for training the ML model.
- `vercel.json`: Configuration for Vercel deployment.
- `.gitignore`: Specifies intentionally untracked files by Git.

## Setup & Running Locally

1.  **Place Data:** Put your `news.csv` (or similar) file into the `/data` directory.
2.  **(Optional but Recommended) Create and activate a virtual environment:**
    ```bash
    # In the project root directory (FakeNewsDetector)
    python -m venv venv
    # On Windows
    .\\venv\\Scripts\\activate
    # On Linux/macOS
    # source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    # Make sure your virtual environment is active
    pip install -r backend/requirements.txt
    ```
4.  **Download NLP Model:** Install the small English model for spaCy:
    ```bash
    python -m spacy download en_core_web_sm
    ```
5.  **Train Classifier Model & Assets:** Run the training script:
    ```bash
    python train_model.py
    ```
    *   This loads your CSVs and creates `models/model_pipeline.pkl`.
6.  **Run Backend:**
    ```bash
    python backend/app.py
    ```
    *   Note: On the first run, the `transformers` library will download the LLM model (e.g., TinyLlama, ~4GB), which may take significant time and disk space. Subsequent runs will use the cached model.
7.  **Access Frontend:** Open your browser to `http://127.0.0.1:5000`.

## Deployment

This project is configured for deployment on Vercel via the `vercel.json` file. "# FakeNewsA1" 
