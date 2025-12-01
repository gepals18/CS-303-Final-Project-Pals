import pandas as pd
from joblib import load

#load saved model+vectorizer
model = load("spam_classifier.joblib")
vectorizer = load("tfidf_vectorizer.joblib")

#loading new emails into dataframe
df = pd.read_excel("email_dataset.xlsx")

#combine subject and message into one column for analysis
df["text"] = df["Subject"].fillna("") + " " + df["Message"].fillna("")

#use vectorizer
x_new = vectorizer.transform(df["text"])

#predict spam/not spam
df["Predicted_Spam"] = model.predict(x_new)
df["Predicted_Label"] = df["Predicted_Spam"].map({0: "Safe Email", 1: "Phishing Email"})
df["Spam_Probability"] = model.predict_proba(x_new)[:, 1]
df.to_excel("emails_with_predictions.xlsx", index=False)
print("Spam prediction finished, results saved to emails_with_predictions.xlsx")