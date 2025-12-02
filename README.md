train_classifier.py: This program needs to be run first, in order to create the ML model and train it with training_emails.xlsx. This is the dataset with 1000 emails (subject and message), 490 phishing and 510 safe.
Running this program will create tfidf_vectorizer.joblib, and spam_classifier.joblib so that parallel processing of emails can occur.
predict_spam.py: This program is run anytime after the model is created and trained. It's used to check email data sets with the trained model and give a report titled emails_with_predictions.xlsx.
I used email_dataset.xlsx to determine the function of predict_spam.py, and you will see those same emails in emails_with_predictions.xlsx plus a percentage that represents the likelyhood that email is spam.
emails_to_check.xlsx is not a dataset that was given to my model. This represents about half of the original dataset from University of Twente, which I discuss this datasets flaws in my paper titled "Final Project Essay Pals.docx"
