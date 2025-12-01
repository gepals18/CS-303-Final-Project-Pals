import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump

#loading training email excel into dataframe
df = pd.read_excel("training_emails.xlsx")

#combine subject and message into one column for analysis
df["text"] = df["Subject"].fillna("") + " " + df["Message"].fillna("")

#convert label text into 1's and 0's
df["label"] = df["Label"].map({"Safe Email": 0, "Phishing Email": 1})

#extract text and spam labels
x = df["text"]
y = df["label"]

#split emails 80/20 for training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=18
)

#convert text to numbers (frequency of appearance)
    #5000 or 2000?
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

#train spam classifier
model = LogisticRegression(max_iter=2000)
model.fit(x_train_tfidf, y_train)

#print model data
y_pred = model.predict(x_test_tfidf)
print(classification_report(y_test, y_pred))

#save model+vectorizer
dump(model, "spam_classifier.joblib")
dump(vectorizer, "tfidf_vectorizer.joblib")

print("training complete, model saved.")