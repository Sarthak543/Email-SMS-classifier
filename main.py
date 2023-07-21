import streamlit as st
import string
import nltk
from nltk.stem.porter import PorterStemmer
import stopwords
import pickle

#  loading vectorizer and model object
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

#  creating function for data preprocessing
# preprocessing function
def transform_text(text):
    text=text.lower()            # converting into lower
    text=nltk.word_tokenize(text)            # converting into words
    
    y=[]            # creating a list and removing the symbols from the text and returning it
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:                #removing all the punctuation and stopwords
        if i not in stopwords.get_stopwords('english') and i not in string.punctuation:
            y.append(i)
            
    ps=PorterStemmer()           #steming all the words i.e   dance,dancing,danced will convert to dance
    
    text=y[:]
    y.clear()
    
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)



st.title("Email/SMS Spam Classifier")

input_sms=st.text_input("Enter the Message")
if st.button("Predict"):

    # Preprocessing text
    transform_sms=transform_text(input_sms)

    # vectorize
    vector_input=tfidf.transform([transform_sms])

    # prediction

    result=model.predict(vector_input)[0]

    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")    