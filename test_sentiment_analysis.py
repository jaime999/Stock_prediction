from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
import unicodedata
import re
import spacy
import stanza
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
from pysentimiento import create_analyzer
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


NEUTRAL_THRESOLD = 0.8


def eliminar_tildes(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def eliminar_caracteres_especiales(texto):
    return re.sub(r'[^a-zA-Z0-9\s]', '', texto)

def eliminar_numero_inicial(texto):
    return re.sub(r'^\d+\.\s*', '', texto)

def eliminar_primera_expansion(df):
    return df[~df['Headlines'].str.contains("la primera de expansion", case=False, na=False)]


def eliminar_stop_words(texto):
    stanza.download("es")
    stop_words = set(stopwords.words('spanish'))
    
    return ' '.join([palabra for palabra in texto.split() if palabra.lower() not in stop_words])


def apply_stemmer(text):
    ps = PorterStemmer()

    return ' '.join([ps.stem(palabra) for palabra in text.split()])




def apply_stemmer_snowball(text):
    stemmer = SnowballStemmer('spanish')

    return ' '.join([stemmer.stem(palabra) for palabra in text.split()])


def apply_lemmer(text):
    wnl = WordNetLemmatizer()
    return ' '.join([wnl.lemmatize(palabra, pos="v") for palabra in text.split()])


def apply_lemmer_spacy(text):
    nlp = spacy.load('es_core_news_sm')
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])


def apply_lemmer_stanza(text):
    nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')
    doc = nlp(text)
    lemmas = [word.lemma for sentence in doc.sentences for word in sentence.words]
    return ' '.join(lemmas)


def traducir_bloque(bloque):
    traducciones = [f"Traducción de {texto}" for texto in bloque]
    return traducciones


def translated_texts(textos):
    # Cargar modelo de traducción
    model_name = "Helsinki-NLP/opus-mt-es-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tamano_bloque = 100
    for i in range(0, len(df), tamano_bloque):
        bloque = df['text'][i:i+tamano_bloque]
        print(bloque)
        inputs = tokenizer(bloque.tolist(), return_tensors="pt",
                           truncation=True, padding=True)
        translated = model.generate(**inputs)
        traducciones = [tokenizer.decode(
            t, skip_special_tokens=True) for t in translated]
        df.loc[i:i+tamano_bloque-1, 'traduccion'] = traducciones

    df.to_csv('traducciones.csv', index=False)
    
    return traducciones


def calculateFinbertToneSentiment(articles_translated):
    print(articles_translated)
    finbert = BertForSequenceClassification.from_pretrained(
        'yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    batch_size = 100  # Ajusta este valor según la memoria de tu sistema
    outputs = []
    for i in range(0, len(articles_translated), batch_size):
        print(i)
        bloque = articles_translated[i:i+batch_size]
        print(bloque)
        # traducciones_bloque = traducir_bloque(bloque)
        # Función para traducir
        inputs = tokenizer(bloque.tolist(), return_tensors="pt",
                           truncation=True, padding=True)
        outputs_batch = finbert(**inputs)
        outputs.append(outputs_batch)
        del bloque
        torch.cuda.empty_cache()

    all_logits = torch.cat([output.logits for output in outputs], dim=0)
    probabilities = torch.nn.functional.softmax(all_logits, dim=-1)
    # Aplicar el umbral a cada texto
    for i, probs in enumerate(probabilities):
        positive_prob = probs[0].item()
        negative_prob = probs[1].item()
        neutral_prob = probs[2].item()

        if neutral_prob > NEUTRAL_THRESOLD:
            predicted_class = "neutral"
        else:
            # Si la probabilidad de neutral es menor que el umbral, elegir entre positivo y negativo
            if positive_prob > negative_prob:
                predicted_class = "positive"
            else:
                predicted_class = "negative"

        result.loc[i] = [df.loc[i]["text"], predicted_class]
        print(f"Text {i+1}: {articles_translated[i]}")
        print(f"Predicted class: {predicted_class}\n")

    result.to_csv('sentiment_finberttone_english.csv', index=False)

    compareResults(result)


def calculateFinanceBERTSpanish(articles_translated):
    model_name = "bardsai/finance-sentiment-es-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenizar el texto
    inputs = tokenizer(articles_translated, return_tensors="pt",
                       truncation=True, padding=True)

    # Obtener las probabilidades de cada clase
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Obtener las etiquetas de las clases
    labels = ["positive", "negative", "neutral"]

    # Obtener la clase con la mayor probabilidad
    max_prob, max_index = torch.max(probs, dim=1)
    predicted_class = labels[max_index.item()]

    # Obtener la probabilidad de la clase neutra
    positive_prob = probs[0][0].item()
    neutral_prob = probs[0][1].item()
    negative_prob = probs[0][2].item()

    # Aplicar el umbral para la clase neutra
    if neutral_prob >= NEUTRAL_THRESOLD:
        predicted_class = "neutral"

    elif positive_prob > negative_prob:
        predicted_class = "positive"

    else:
        predicted_class = "negative"

    return predicted_class


def calculateRobertuitoSpanish(text):
    print(text)
    analyzer = create_analyzer(task="sentiment", lang="es")
    resultado = analyzer.predict(text)
    # Obtén las probabilidades de las clases
    probas = resultado.probas
    # Clasifica según el umbral
    if probas["NEU"] > NEUTRAL_THRESOLD:
        sentiment = "neutral"
    else:
        if probas["POS"] > probas["NEG"]:
            # Si NEU no supera el umbral, elige la clase con la mayor probabilidad
            sentiment = "positive"

        else:
            sentiment = "negative"

    return sentiment


def calculatexlmrSpanish(text):
    model = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(model)

    model = AutoModelForSequenceClassification.from_pretrained(model)

    encoded_input = tokenizer(
        text, return_tensors='pt', truncation=True, padding=True)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    negative_prob = scores[0]
    neutral_prob = scores[1]
    positive_prob = scores[2]

    # Aplicar el umbral para la clase neutra
    if neutral_prob >= NEUTRAL_THRESOLD:
        predicted_class = "neutral"

    elif positive_prob > negative_prob:
        predicted_class = "positive"

    else:
        predicted_class = "negative"

    return predicted_class


def compareResults(result):
    matches1 = result['target_sentiment'] == df['target_sentiment']
    matches2 = result['target_sentiment'] == df['consumers_sentiment']
    matches3 = result['target_sentiment'] == df['companies_sentiment']

    # Calcular el porcentaje de coincidencia
    percentage_match1 = matches1.mean() * 100
    percentage_match2 = matches2.mean() * 100
    percentage_match3 = matches3.mean() * 100

    print(
        f"El porcentaje de coincidencia de target_sentiment es: {percentage_match1:.2f}%")
    print(
        f"El porcentaje de coincidencia de consumers_sentiment es: {percentage_match2:.2f}%")
    print(
        f"El porcentaje de coincidencia de companies_sentiment es: {percentage_match3:.2f}%")


def create_confusion_matrix(colReal, colPred, predicted_labels=['positive', 'neutral', 'negative']):
    conf_matrix = confusion_matrix(colReal, colPred)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels=predicted_labels)
    
    cm_display.plot()
    plt.show()
    
def get_sentiment_metrics(colReal, colPred):
    # Calcular precisión
    accuracy = accuracy_score(colReal, colPred)
    
    # Calcular f-score con promedio 'weighted'
    f1_score_macro = f1_score(colReal, colPred, average='macro')
    f1_score_weighted = f1_score(colReal, colPred, average='weighted')
    
    print(f"Precisión: {accuracy}")
    print(f"F1-Score Macro: {f1_score_macro}")
    print(f"F1-Score Weighted: {f1_score_weighted}")


def analyze_sentiment_generic(modelName, text, probabilities_pos):
  tokenizer = AutoTokenizer.from_pretrained(modelName)
  model = AutoModelForSequenceClassification.from_pretrained(modelName)

  encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
  output = model(**encoded_input)
  scoresNumpy = output[0][0].detach().numpy()
  scores = softmax(scoresNumpy)

  # Print labels and scores
  negative_prob = scores[probabilities_pos['negative']]
  neutral_prob = scores[probabilities_pos['neutral']]
  positive_prob = scores[probabilities_pos['positive']]
  # Aplicar el umbral a la clase neutra
  if neutral_prob > NEUTRAL_THRESOLD:
        sentiment = "neutral"
  else:
        # Si no es neutro, determinar entre positivo y negativo
        if positive_prob > negative_prob:
            sentiment = "positive"
        else:
            sentiment = "negative"

  return sentiment

def calculateNewsSentimentBERT(headlines, tokenizer, model):
    # texts = [headline['heading'] for headline in headlines]
    texts = headlines
    inputs = tokenizer(texts, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits

    return int(torch.argmax(logits))


def analyze_sentiment_mbert(text):
    modelName = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    sentiment = calculateNewsSentimentBERT(text, tokenizer, model)
    if sentiment < 2:
        return "negative"

    elif sentiment > 2:
        return "positive"

    return "neutral"


result = pd.DataFrame(columns=["Headlines", "target_sentiment"])
# df = pd.read_csv('FinancES_phase_2_train_public.csv')
df = pd.read_csv('traducciones.csv')
df['target_sentiment'] = df['target_sentiment'].str.replace(
    'postive', 'positive')
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(eliminar_tildes)
df['text'] = df['text'].apply(eliminar_caracteres_especiales)

df['model_sentiment_xlmr'] = df['text'].apply(lambda x: analyze_sentiment_generic(
    'cardiffnlp/twitter-xlm-roberta-base-sentiment', x,
    {'negative': 0, 'neutral': 1, 'positive': 2}))

df['model_sentiment_finance_es'] = df['text'].apply(lambda x: analyze_sentiment_generic(
    'bardsai/finance-sentiment-es-base', x,
    {'negative': 2, 'neutral': 1, 'positive': 0}))

df['model_sentiment_finbert'] = df['traduccion'].apply(lambda x: analyze_sentiment_generic(
    'yiyanghkust/finbert-tone', x,
    {'negative': 2, 'neutral': 0, 'positive': 1}))

df['model_sentiment_mbert'] = df['text'].apply(lambda x: analyze_sentiment_mbert(x))
