# END TO END text_summarizer PTOJECT

## WORKFLOWS

1.UPDATE CONFIG.PY
2.UPDATE PARAMS.YAML
3.UPDATE ENTITY
4.UPDATE THE CONFIGURATION MANAGER IN SRC , CONFIG
5.UPDATE HE COMPONENTS
6.UPDATE THE PIPELINE
7.UPDATE MAIN.PY
8.UPDATE APP.PY


END-TO-END TEXT SUMMARIZATION PROJECT

An end-to-end NLP Text Summarization system built using Python, Hugging Face Transformers, FastAPI, Docker, and MLOps pipeline architecture.

This project takes long paragraphs or articles as input and generates concise summaries using a fine-tuned T5 Transformer model.

PROJECT OVERVIEW

Text summarization is one of the most important Natural Language Processing (NLP) tasks.
This project focuses on building a complete production-style pipeline for abstractive text summarization.

The application:

Accepts long text input
Processes text using a trained Transformer model
Generates meaningful summaries
Exposes prediction APIs using FastAPI
Supports deployment using Docker and Hugging Face Spaces
PROJECT ARCHITECTURE
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Prediction Pipeline
      ↓
FastAPI Deployment
TECH STACK
Programming Language
Python
Libraries & Frameworks
Hugging Face Transformers
PyTorch
FastAPI
Uvicorn
YAML
NumPy
Pandas
MLOps Concepts Used
Modular Pipeline Architecture
Configuration Management
Training Pipelines
Model Evaluation
API Deployment
Dockerization
FEATURES
End-to-end NLP pipeline
Transformer-based text summarization
FastAPI REST API
Swagger UI testing
Docker support
Hugging Face deployment ready
Modular and scalable code structure
Config-driven architecture
PROJECT STRUCTURE
text_summarizer/
│
├── artifacts/
│   └── model_trainer/
│       ├── tokenizer/
│       └── t5-small-model/
│
├── src/
│   └── textSummarizer/
│       ├── pipeline/
│       ├── utils/
│       └── components/
│
├── app.py
├── main.py
├── params.yaml
├── requirements.txt
├── Dockerfile
├── setup.py
└── README.md
WORKFLOW
1. Data Ingestion

Downloads and loads the dataset required for training.

2. Data Validation

Checks dataset quality and validates required files.

3. Data Transformation

Tokenizes and preprocesses the dataset for model training.

4. Model Training

Fine-tunes the T5 Transformer model for summarization.

5. Model Evaluation

Evaluates model performance using NLP evaluation metrics.

6. Prediction Pipeline

Loads trained tokenizer and model to generate summaries.

7. API Deployment

FastAPI serves prediction endpoints for real-time inference.

MODEL USED
T5 Transformer Model

This project uses:

t5-small

T5 (Text-To-Text Transfer Transformer) converts every NLP task into a text generation problem.

Example:

Input:
summarize: Artificial Intelligence is transforming industries...

Output:
AI is transforming industries through automation and innovation.
API ENDPOINTS
Train Model
GET /train

Triggers the model training pipeline.

Predict Summary
POST /predict
Request Body
{
  "text": "Your long paragraph here..."
}
Response
{
  "summary": "Generated summarized text."
}
RUN PROJECT LOCALLY
1. Clone Repository
git clone https://github.com/Riyazzshaik/text_summarizer.git
2. Create Virtual Environment
conda create -n texts python=3.10 -y
conda activate texts
3. Install Dependencies
pip install -r requirements.txt
4. Run Application
python app.py
FASTAPI DOCUMENTATION

After running the server:

http://localhost:8080/docs

Swagger UI allows testing APIs directly from the browser.

DOCKER SUPPORT
Build Docker Image
docker build -t text-summarizer .
Run Container
docker run -p 8080:8080 text-summarizer
HUGGING FACE DEPLOYMENT

This project can also be deployed using:

Hugging Face Spaces
Docker SDK
FastAPI
SAMPLE TEST INPUT
Artificial Intelligence is transforming industries around the world. Companies are using AI to automate repetitive tasks, improve customer service, and analyze huge amounts of data.
Generated Summary
AI is transforming industries through automation and data analysis.
LEARNING OUTCOMES

Through this project, I learned:

NLP pipeline development
Transformer model training
Hugging Face ecosystem
FastAPI deployment
Docker containerization
Model inference pipelines
MLOps workflow structuring
FUTURE IMPROVEMENTS
Add React frontend
Improve summarization quality
Deploy on cloud platforms
Add authentication
Support multilingual summarization
Add GPU optimization
Integrate CI/CD pipelines
AUTHOR
Riyaz Shaik

Machine Learning & AI Enthusiast
Focused on NLP, MLOps, and Full Stack AI Applications

GitHub:
Riyaz Shaik GitHub

LICENSE

This project is created for educational and learning purposes.
