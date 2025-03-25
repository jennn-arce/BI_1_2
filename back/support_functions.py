import os
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.snowball import SnowballStemmer
from num2words import num2words
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from pydantic import BaseModel, Field
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from fastapi import FastAPI, HTTPException
# Importaciones generales
import sys
import re
import string
import unicodedata
import numpy as np
import pandas as pd
# Procesamiento de lenguaje natural (NLP)
import nltk
from nltk import word_tokenize, sent_tokenize, PorterStemmer
from nltk.corpus import stopwords

# Descarga de recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración de stopwords y tokenizador
stop_words = stopwords.words('spanish')
wpt = nltk.WordPunctTokenizer()
ps = PorterStemmer()


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
            new_word = unicodedata.normalize('NFKD', word).encode(
                'ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
    return new_words


def normalize_text(text, repetar_español=False):
    if not repetar_español:
        text = text.replace('á', 'a')
        text = text.replace('é', 'e')
        text = text.replace('í', 'i')
        text = text.replace('ó', 'o')
        text = text.replace('ú', 'u')
        text = text.replace('ü', 'u')
        text = text.replace('ñ', 'n')

    # Eliminate punctuation and special characters by replacing them
    text = text.replace('(', '').replace(')', '')
    text = text.replace('[', '').replace(']', '')
    text = text.replace('{', '').replace('}', '')
    text = text.replace('<', '').replace('>', '')
    return text.lower()


def to_lowercase(words, respetar_español=False):
    return [normalize_text(word, respetar_español) for word in words]


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = num2words(word, lang='es')
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    nuevas_palabras = []
    for palabra in words:
        if palabra is not None:
            if palabra not in stopwords.words('spanish'):
                nuevas_palabras.append(palabra)
    return nuevas_palabras


def corregir_contracciones_espanol(texto):
    texto = texto.replace('al ', 'a el ').replace(
        'al.', 'a el.')  # "al" a "a el"
    # "del" a "de el"    # Agrega más reglas aquí según sea necesario
    texto = texto.replace('del ', 'de el ').replace('del.', 'de el.')
    return texto


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = SnowballStemmer("spanish")
    return [stemmer.stem(word) for word in words]


def lemmatize_verbs(words):
    """Simple lemmatization for verbs in list of tokenized words"""
    # Simple rule-based lemmatization (you can extend this as needed)
    lemmatized_words = []
    for word in words:
        if word.endswith('ar'):
            lemmatized_words.append(word[:-2])  # Remove 'ar' (basic rule)
        elif word.endswith('er'):
            lemmatized_words.append(word[:-2])  # Remove 'er' (basic rule)
        elif word.endswith('ir'):
            lemmatized_words.append(word[:-2])  # Remove 'ir' (basic rule)
        else:
            # Return the word as is if no rule applies
            lemmatized_words.append(word)
    return lemmatized_words


def stem_and_lemmatize(words, stems_parameter=True, lemmas_parameter=True):
    """Stem and Lemmatize words"""
    if stems_parameter and lemmas_parameter:
        stems = stem_words(words)
        lemmas = lemmatize_verbs(words)
        return stems + lemmas
    elif lemmas_parameter:
        lemmas = lemmatize_verbs(words)
        return lemmas
    elif stems_parameter:
        stems = stem_words(words)
        return stems
    else:
        return words


def preprocessing(words, respetar_español=False, respetar_ascii=False):
    words = words.split()
    words = to_lowercase(words, respetar_español)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    if not respetar_ascii:
        words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words


def transfor_data_local(df):
    columnas = ['Titulo', 'Descripcion']
    df = df[columnas]
    # fill na with ''
    df = df.fillna('')
    for columna in columnas:
        df[columna] = df[columna].apply(corregir_contracciones_espanol)
        df[columna] = df[columna].apply(
            lambda x: preprocessing(x, False, True))
        df[columna] = df[columna].apply(
            lambda x: stem_and_lemmatize(x, False, True))
        df[columna] = df[columna].apply(lambda x: ' '.join(map(str, x)))
    return df


def createModel(text_transformer, df):
    df = df.fillna('')
    df = df[~((df['Titulo'] == '') & (
        df['Descripcion'] == '') & (df['Label'] == ''))]
    X = df[['Titulo', 'Descripcion']]
    y = df['Label']
    data_transformer = FunctionTransformer(transfor_data_local)

    # Apply TfidfVectorizer separately to 'Titulo' and 'Descripcion'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10
    )

    pipeline = Pipeline([
        ("data_transform", data_transformer),  # Preprocess the dataframe
        ("vectorizer", text_transformer),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    results = classification_report(y_test, y_pred, output_dict=True)

    return results, pipeline


def loadModel(MODEL_PATH):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Archivo del modelo no encontrado en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)
