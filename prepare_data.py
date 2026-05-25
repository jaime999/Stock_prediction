from economic_math import getRSI
from pysentimiento import create_analyzer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from test_sentiment_analysis import eliminar_tildes, eliminar_caracteres_especiales, eliminar_primera_expansion

def get_sentiment_probs(text):
    analyzer = create_analyzer(task="sentiment", lang="es")
    result = analyzer.predict(text)
    
    return result.probas


def merge_articles_economic(articles):
    sentiment_probs = articles['Headlines'].apply(get_sentiment_probs)
    sentiment_df = pd.DataFrame(sentiment_probs.tolist(), index=articles.index)
    articles = pd.concat([articles, sentiment_df], axis=1)

    print(articles)
    articles = articles.groupby('Fecha', as_index=False)[
        ['POS', 'NEU', 'NEG']].mean()

    csvFilePath = 'HistoricDataBBVA_12-01-24_12-01-25.csv'
    dfBBVA = pd.read_csv(csvFilePath, decimal=',')
    dfBBVA['Fecha'] = pd.to_datetime(dfBBVA['Fecha'], format='%d.%m.%Y')
    articles['Fecha'] = pd.to_datetime(dfBBVA['Fecha'])
    dfBBVA = dfBBVA.sort_values(by='Fecha')

    print(dfBBVA)

    dfMerged = pd.merge(dfBBVA, articles, on='Fecha', how='left')
    dfMerged['RSI'] = getRSI(dfMerged['Último'])
    dfMerged = dfMerged.dropna(subset=['RSI'])

    dfMerged = dfMerged.drop(columns=['Vol.', '% var.'])
    dfMerged = dfMerged.fillna(0)

    print(dfMerged)
    return dfMerged


def get_economic_data(csvFilePath, bank):
    dfBBVA = pd.read_csv(csvFilePath, decimal=',')
    dfBBVA['Fecha'] = pd.to_datetime(dfBBVA['Fecha'], format='%d.%m.%Y')
    dfBBVA = dfBBVA.sort_values(by='Fecha')

    dfBBVA['RSI'] = getRSI(dfBBVA['Último'])
    dfBBVA = dfBBVA.dropna(subset=['RSI'])

    dfBBVA = dfBBVA.drop(columns=['Vol.', '% var.'])

    dfBBVA.columns = [
        col if col == 'Fecha' else f"{col}_{bank}" for i, col in enumerate(dfBBVA.columns)]
    print(dfBBVA)

    return dfBBVA


def analyze_sentiment_generic(modelName, articles, textCol, probabilities_pos, colName):
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    
    tamano_bloque = 100
    for i in range(0, len(articles), tamano_bloque):
        bloque = articles[textCol][i:i+tamano_bloque]
        print(bloque)
        
        encoded_input = tokenizer(bloque.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
        output = model(**encoded_input)
        scores = [softmax(output[0][i].detach().numpy()) for i in range(len(bloque))]
                
        positive = [arr[probabilities_pos['positive']] for arr in scores]
        neutral = [arr[probabilities_pos['neutral']] for arr in scores]
        negative = [arr[probabilities_pos['negative']] for arr in scores]

        articles.loc[i:i+tamano_bloque-1, f'{colName}_positive'] = positive
        articles.loc[i:i+tamano_bloque-1, f'{colName}_neutral'] = neutral
        articles.loc[i:i+tamano_bloque-1, f'{colName}_negative'] = negative

    return articles

def calculateNewsSentimentBERT(headlines, tokenizer, model):
    texts = headlines
    inputs = tokenizer(texts, return_tensors="pt",
                        padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    scores = logits.softmax(dim=1)
    averageScore = scores.mean(dim=0).tolist()

    return averageScore

def analyze_sentiment_mbert(articles):
    modelName = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    tamano_bloque = 100
    for i in range(0, len(articles), tamano_bloque):
        bloque = articles['Headlines'][i:i+tamano_bloque]
        print(bloque)
        
        encoded_input = tokenizer(bloque.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)
        output = model(**encoded_input)
        scores = [softmax(output[0][i].detach().numpy()) for i in range(len(bloque))]
                
        oneStars = [arr[0] for arr in scores]
        twoStars = [arr[1] for arr in scores]
        threeStars = [arr[2] for arr in scores]
        fourStars = [arr[3] for arr in scores]
        fiveStars = [arr[4] for arr in scores]

        articles.loc[i:i+tamano_bloque-1, 'mbert_oneStars'] = oneStars
        articles.loc[i:i+tamano_bloque-1, 'mbert_twoStars'] = twoStars
        articles.loc[i:i+tamano_bloque-1, 'mbert_threeStars'] = threeStars
        articles.loc[i:i+tamano_bloque-1, 'mbert_fourStars'] = fourStars
        articles.loc[i:i+tamano_bloque-1, 'mbert_fiveStars'] = fiveStars

    return articles
    
def analyze_sentiment_pysentimiento(articles):
    analyzer = create_analyzer(task="sentiment", lang="es")
    
    tamano_bloque = 100
    for i in range(0, len(articles), tamano_bloque):
        bloque = articles['Headlines'][i:i+tamano_bloque]
        print(bloque)
        
        result = analyzer.predict(bloque)
        positive = [arr.probas['POS'] for arr in result]
        neutral = [arr.probas['NEU'] for arr in result]
        negative = [arr.probas['NEG'] for arr in result]
        
        print(positive)
                
        articles.loc[i:i+tamano_bloque-1, 'pysentimiento_positive'] = positive
        articles.loc[i:i+tamano_bloque-1, 'pysentimiento_neutral'] = neutral
        articles.loc[i:i+tamano_bloque-1, 'pysentimiento_negative'] = negative

    return articles


def get_generic_sentiment(articles):
    articles['Headlines'] = articles['Headlines'].str.lower()
    articles['Headlines'] = articles['Headlines'].apply(eliminar_tildes)
    articles['Headlines'] = articles['Headlines'].apply(eliminar_caracteres_especiales)
    articles['traduccion'] = articles['traduccion'].str.lower()
    articles['traduccion'] = articles['traduccion'].apply(eliminar_tildes)
    articles['traduccion'] = articles['traduccion'].apply(eliminar_caracteres_especiales)

    sentiment_probs = analyze_sentiment_generic('cardiffnlp/twitter-xlm-roberta-base-sentiment',
                                                articles, 'Headlines', {'negative': 0, 'neutral': 1, 'positive': 2},
                                                'xlmr')
    sentiment_probs = analyze_sentiment_generic('bardsai/finance-sentiment-es-base',
                                                articles, 'Headlines', {'negative': 2, 'neutral': 1, 'positive': 0},
                                                'finance-sentiment-es')
    sentiment_probs = analyze_sentiment_mbert(articles)
    sentiment_probs = analyze_sentiment_generic('yiyanghkust/finbert-tone',
                                                articles, 'traduccion', {'negative': 2, 'neutral': 0, 'positive': 1},
                                                'finbert')
    sentiment_probs = analyze_sentiment_pysentimiento(articles)
    
    sentiment_probs = eliminar_primera_expansion(sentiment_probs)
    
    return sentiment_probs
