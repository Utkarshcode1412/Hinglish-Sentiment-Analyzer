import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Load the dataset from the CSV file
print("Loading data from data/reviews.csv...")
try:
    df = pd.read_csv("data/reviews.csv")
except FileNotFoundError:
    print("Error: Could not find 'data/reviews.csv'. Please check your folder structure.")
    exit()

# 2. Split the dataset into training and testing
# (This looks for the 'text' and 'sentiment' columns in your CSV)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.25, random_state=42)

# 3. Create and Train the Model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
print("Training the AI model... Done!\n")

# ==========================================
# INTERACTIVE CUSTOM INPUT & OUTPUT SECTION
# ==========================================

print("-" * 50)
print("Welcome to the Hinglish Sentiment Analyzer!")
print("Type a sentence to see if it's Positive, Negative, or Neutral.")
print("Type 'exit' or 'quit' to stop the program.")
print("-" * 50)

while True:
    user_text = input("\nEnter your sentence: ")
    
    if user_text.lower() in ['exit', 'quit']:
        print("Exiting the analyzer. Phir milenge!")
        break
        
    if not user_text.strip():
        print("Please type something before pressing Enter.")
        continue
        
    predicted_sentiment = model.predict([user_text])[0]
    print(f"--> AI Prediction: {predicted_sentiment}")
