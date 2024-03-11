import pickle

# Load the model from the file object
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
    
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
    

# model = pickle.load("logistic_regression_model.pkl")
# le = pickle.load("label_encoder.pkl")
# tfidf = pickle.load("tfidf.pkl")


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()                                                            # Lowercase text
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])     # Remove punctuation and numbers
    stop_words = set(stopwords.words('english'))                                    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    text = ' '.join([lem.lemmatize(word) for word in text.split()])                 # Lemmatize words 
    text = text.strip()                                                             # removes leading and trailing whitespaces

    return text


import streamlit as st

def main():
    """
    This function creates a Streamlit app with a text field and a submit button.
    """

    with st.form("my_form"):
        text_input = st.text_input(label="Enter The Poem:", key="text_input")
        submitted = st.form_submit_button(label="Find The Genre")

    if submitted:
        
        # preprocessing the text
        processed_text = preprocess_text(text_input)
        
        # Vecrorizing text
        vec_text = tfidf.transform([processed_text])
        
        # Predicting the Genre
        pred_ = model.predict(vec_text)
        
        # Inversing the result 
        genre = le.inverse_transform(pred_)
        
        # Process the entered text here
        st.write(f"The Poem is of Genre: **{genre[0]}**")

if __name__ == "__main__":
    main()


