from bs4 import BeautifulSoup
import requests
from datetime import datetime
import time
import re

url = '//cgi.expansion.com/buscador/archivo_expansion.html?fecha_busq_avanzada=1&diaDesde=01&mesDesde=01&anyoDesde=2020&diaHasta=31&mesHasta=10&anyoHasta=2025&n=100&w=60&w=1&q=bbva+sabadell&buscar=Buscar'
url2 = 'https://cgi.expansion.com/buscador/archivo_expansion.html?q=bbva%20sabadell&t=1&i=101&n=100&fd=28/01/2024&td=31/10/2025&w=60&s=1&fecha_busq_avanzada=1'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

def getScrappingNews(newsUrl):

    urlPage = f'https:{newsUrl}'
    # Obtener el contenido de la página
    response = requests.get(urlPage, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # Encontrar todos los elementos con la clase 'articleHeadline'
    contenedor = soup.find_all(class_='detalle_noticia_busqueda')
    articlesTextAux = []
    articlesDateAux = []

    # Obtener el texto interno de cada elemento
    for index, headline in enumerate(contenedor):
        etiqueta_a = headline.find('a')  # Encuentra la etiqueta <a> dentro del contenedor
        articlesTextAux.append(etiqueta_a.get_text())
        clase_firma = headline.find(class_='firma')
        articlesDateAux.append(clase_firma.get_text())

    article_dates = soup.find_all(class_="articleTime")
    for articleDate in article_dates:
        try:
            datetime.strptime(articleDate.get_text(), '%d/%m')
            articlesDateAux.append(articleDate.get_text())

        except ValueError:
            continue

    next_page = soup.find_all(class_="siguiente")
    if len(next_page) > 0:
        url_next_page = soup.find_all(class_="siguiente")[1]['href']
        
    else:
        url_next_page = None

    return articlesTextAux, articlesDateAux, url_next_page


def get_articles_dates():
    end_exec = False
    urlPage = url
    articlesText = []
    articlesDate = []
    while not end_exec:
        articlesTextAux, articlesDateAux, urlPage = getScrappingNews(urlPage)
        time.sleep(5)
        if len(articlesText) == 0:
            articlesText = articlesTextAux
            articlesDate = articlesDateAux
    
        else:
            articlesText = articlesText + articlesTextAux
            articlesDate = articlesDate + articlesDateAux
        
        print(urlPage)
        if not urlPage:
            end_exec = True
            
    return articlesText, articlesDate

def get_stock_price_bbva():
    bbvaURL = 'https://www.investing.com/equities/bbva-historical-data'
    response = requests.get(bbvaURL, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # Compilar una expresión regular para buscar el patrón deseado
    pattern = re.compile(r'BBVA Stock Price History\.csv')
    
    # Buscar la etiqueta <a> cuyo atributo download contenga el patrón
    a_tag = soup.find('a', attrs={'download': pattern})
    
    # Imprimir la etiqueta encontrada
    print(a_tag)