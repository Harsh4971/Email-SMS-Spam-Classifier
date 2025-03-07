import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize
print(word_tokenize("This is a test sentence."))

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()     #Converting to lowercase
    text = nltk.word_tokenize(text)     #Separating all words

    y = []                  #Removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:  # Removing stopwords and punctuations (Stopwords are normal words that don't carry meaning)
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:  # Considering words like dance or dancing or danced to be same is called stemming
        y.append(ps.stem(i))

    return " ".join(y)   #Converting all to string

st.title("Email/SMS SPam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    #Preprocessing
    transformed_sms = transform_text(input_sms)

    #Vectorize
    vector_input = tfidf.transform([transformed_sms])

    #Predict
    result = model.predict(vector_input)[0]

    #Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
