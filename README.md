# Spam Classifier

A small machine learning project to classify SMS messages as **spam** or **ham** (legitimate messages).

## About

This project was built as a first NLP and machine learning experiment. 
It helped me understand how text data can be transformed into numbers and used to train a model.

The model uses **TF-IDF vectorization** and **logistic regression** to detect spam messages with around **98% accuracy**.

## Dataset

The project uses the **SMS Spam Collection** dataset from Kaggle/UCI.  
It contains about **5,500 SMS messages** labeled as spam or ham.

## How it works

The pipeline is simple:
- Load and clean the data with pandas  
- Split into training and test sets  
- Vectorize the text using TF-IDF  
- Train a logistic regression model with **balanced class weights**  
- Evaluate performance on the test set  

Class balancing is used because spam messages are much less common than regular messages in the dataset.

## Results

On the test set:
- **Accuracy:** ~98%  
- **Spam recall:** ~91%  

The model does a good job at catching spam while keeping false positives low.

## Requirements

```bash
pip install pandas numpy scikit-learn
```

## Usage

Put `spam.csv` in the same folder as `spam.py` and run:

```bash
python3 spam.py
```

After training, you can type a message in the terminal to see if it is spam or not.
Type `q` to quit.

## Project structure

```
spam-classifier-ml/
├── spam.py
├── spam.csv
└── README.md
```

## What I learned

This was my first NLP project and I learned a lot about:
- Text preprocessing
- TF-IDF vectorization
- Handling imbalanced datasets
- Evaluating a classification model

Building the interactive CLI also helped me understand how a model can be used in practice.

## Possible improvements

- Experiment with **n-grams** to capture word patterns better
- Try other algorithms like **Naive Bayes** or **SVM**
- Save the trained model so it does not need to be retrained every time
- Turn the project into a simple **web application**