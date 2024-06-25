import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pickle


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]

    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]

    y.clear()

    ps = PorterStemmer()

    for i in text:  # Corrected the syntax here
        y.append(ps.stem(i))

    return " ".join(y)


Tfid = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open("model.pkl", "rb"))

st.title("Data Mining Project")

input_sms = st.text_input("Enter Message Here")


if st.button("Submit"):
    transformed_Text = transform_text(input_sms)
    vector_Text = Tfid.transform([transformed_Text])
    Result = model.predict(vector_Text)[0]
    if Result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
