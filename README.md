```
# Mental Health Chatbot using Gemini Pro API

This project demonstrates how to build a mental health chatbot using Google's Gemini Pro API, with the MentalChat16K dataset for benchmarking.

---

## 📦 Step 1: Set Up Development Environment

Install required Python packages:

```
!pip install google-generativeai pandas datasets rouge-score
!pip install langchain langchain_community
!pip install -q google-generativeai
```

Import libraries:

```python
import google.generativeai as genai
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge
from langchain_community.chat_models import ChatOpenAI
```

---

## 🔑 Step 2: Access Gemini Pro API

Set your API key:

```python
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=GOOGLE_API_KEY)
```

Check available models:

```python
available_models = genai.list_models()
for model in available_models:
    print(model.name)
```

Test the Gemini API:

```python
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
response = model.generate_content("How can I manage anxiety?")
print(response.text)
```

---

## 🧠 Step 3: Load & Preprocess MentalChat16K Dataset

Download the dataset:

```python
dataset = load_dataset("ShenLab/MentalChat16K")
df = pd.DataFrame(dataset["train"])
```

Clean the dataset:

```python
df.drop_duplicates(inplace=True)
df = df[['input', 'output']]
df.to_csv("mentalchat16k_cleaned.csv", index=False)
```

---

## 🤖 Step 4: Implement the Chatbot

Define chatbot function:

```python
def chat_with_gemini(user_input):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(user_input)
    return response.text
```

Test it:

```python
print(chat_with_gemini("How can I manage anxiety?"))
```

---

## 🧪 Step 5: Hyperparameter Tuning

### 5.1 Tune Generation Settings

```python
response = model.generate_content(
    "How can I manage anxiety?",
    generation_config={
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "max_output_tokens": 200
    }
)
```

### 5.2 Use Prompt Engineering

```python
prompt = """You are a supportive mental health assistant.
A user is feeling anxious and needs guidance. Offer practical, empathetic advice."""
response = model.generate_content(prompt)
```

---

## 📏 Step 6: Evaluation Metrics

Install evaluation libraries:

```
!pip install bert-score nltk
```

Metrics used:
- **ROUGE-1**: unigram overlap
- **ROUGE-2**: bigram overlap
- **ROUGE-L**: longest common subsequence
- **BLEU**: n-gram precision
- **BERTScore**: semantic similarity using transformers

---

## ✅ Summary

This notebook provides a baseline pipeline for building and evaluating a mental health-focused chatbot using Gemini Pro. It includes:
- Dataset preprocessing
- Prompt engineering
- Hyperparameter tuning
- Automatic evaluation with NLP metrics

Replace `"YOUR_API_KEY_HERE"` with your actual API key before running.
```
