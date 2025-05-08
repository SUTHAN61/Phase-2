import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load datasets
fake_df = pd.read_csv(r"C:\Users\prince\Desktop\suuu\fake_news_dataset.csv")
real_df = pd.read_csv(r"C:\Users\prince\Desktop\suuu\true_news_dataset.csv")

# Combine datasets
combined_df = pd.concat([fake_df, real_df], ignore_index=True)

# Preprocessing
X = combined_df["text"]
y = combined_df["label"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english', max_df=0.7)),
    ("clf", LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# GUI setup
def predict_news():
    user_input = text_input.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Input Required", "Please enter news text to predict.")
        return
    prediction = model.predict([user_input])[0]
    result = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    result_label.config(text=f"Prediction: {result}", fg="red" if prediction == 1 else "green")

root = tk.Tk()
root.title("Fake News Detector")

# GUI layout
tk.Label(root, text="Enter News Text:").pack(pady=5)
text_input = tk.Text(root, height=10, width=60)
text_input.pack(pady=5)
tk.Button(root, text="Predict", command=predict_news).pack(pady=5)
result_label = tk.Label(root, text="Prediction:", font=("Helvetica", 14))
result_label.pack(pady=10)

root.mainloop()
