#!/usr/bin/env python
# coding: utf-8

# # <center>Challenge Etermax | Cecilia Trejo </center>

# #### Objetivo 
# Determinar los tópicos predominantes en cada conversación del corpus adjunto. 
# 
# #### Notas:
# - Se adjunta corpus (en formato csv) compuesto por la desgrabación de 640 llamadas.
# - Cada desgrabación corresponde a un llamado en el que interviene un cliente y un agente de un contact center.
# 
# #### Lineamientos:
# 
# - El lenguaje a utilizar es Python, y se debe entregar una Notebook en donde se puedan visualizar todos los pasos ejecutados, desde la carga del corpus, hasta la predicción de cada conversación realizada por el modelo.
# - Se pueden utilizar las librerías y técnicas que se consideren, no hay limitación en cuanto a este punto.
# - El código deberá ser compartido a través de una cuenta de Git.
# 

# #### Instalación de Librerías (comandos a ejecutar a través de la consola)

#!pip install spacy
#!pip install pandas 
#!pip install nltk 
#!pip install gensim 
#!python -m spacy download es_core_news_sm


# #### Importación de Librerías 
# En esta sección, se importan todas las librerías necesarias para realizar el procesamiento y modelado de tópicos. Estas bibliotecas incluyen Gensim para el modelado de tópicos, spaCy para el procesamiento de texto, nltk para la gestión de stopwords. 



# Gensim
import gensim
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel

# Manipulación de Archivos
import pandas as pd
import os

# Spacy
import spacy

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Se descargan los datos necesarios para el tokenizador de NLTK.
nltk.download('punkt') 

# Se descarga las stopwords en español y se almacenan en la variable stop_words.
nltk.download('stopwords')  

# Se descarga el lematizador de NLTK y se almacena en la variable lemmatizer
nltk.download('wordnet')


# #### Versiones de librerías
print("Versión de gensim:", gensim.__version__) 
print("Versión de Spacy:", spacy.__version__) 
print("Versión de NLTK:", nltk.__version__) 
print("Versión de Pandas:", pd.__version__) 


# #### Carga del DataFrame:
# Establece la variable folder como la ubicación de los archivos de datos alojados en la carpeta "./datasets" y realiza una lista de los archivos que contiene la carpeta folder.
folder = "./datasets"

# Se listan los archivos del path folder
os.listdir(folder)

# Se crea un DataFrame vacío en la variable 'df' utilizando la librería Pandas.
df = pd.DataFrame()

# Se crea un ciclo for que itera a través de los archivos de la folder 
for file in os.listdir(folder):
    print("File = ", file)
    
    # Se concatena folder con el file en cuestión para construir la ruta completa del archivo
    file_path = os.path.join(folder, file)
    print("File path = ", file_path)
    
    # Se lee el archivo CSV en un DataFrame. Se utiliza como parámetro de delimitación la pleca (|) para que las interacciones se alineen en una sola columna.
    file_df = pd.read_csv(file_path, sep="|")
    
    # Se agrega una columna al DataFrame que indica de qué archivo es esta data
    file_df["file_path"] = file_path
    
    # Se concatena file_df al DataFrame principal que antes estaba vacío
    df = pd.concat([df, file_df], ignore_index=True)


# #### DataFrame
# En este print se puede observar que el DataFrame ya no está vacio y que aparece la nueva columna file_path que cree en la celda anterior
df

# #### Preprocesamiento del texto: 
# Esto incluye tokenización, eliminación de stopwords y lematización. 
# Se utiliza NLTK y Spacy con el modelo de lenguaje en español ("es_core_news_sm") para realizar la lematización de las conversaciones.


# Se convierten las palabras a minúsculas y luego se tokeniza
def lower_and_tokenize(text):
    return word_tokenize(text.lower())
df['Pregunta'] = df['Pregunta'].apply(lower_and_tokenize)

stop_words = set(stopwords.words('spanish'))  

# Se analiza palabra por palabra del DataFrame y se filtran aquellas que no son stopwords
def remove_stopwords(text):
    return [word for word in text if word not in stop_words]
df['Pregunta'] = df['Pregunta'].apply(remove_stopwords)

# Se realiza un preprocesamiento utilizando la función simple_preprocess de Gensim para la eliminación de acentos y caracteres no alfabéticos.
def preprocess(text):
    return [word for word in simple_preprocess(' '.join(text), deacc=True)]
df['Pregunta'] = df['Pregunta'].apply(preprocess)

lemmatizer = WordNetLemmatizer()

def lemmatize_words(word_list):
    return [lemmatizer.lemmatize(word) for word in word_list]

# Se lematiza cada lista de palabras en 'Pregunta'
df['Pregunta'] = df['Pregunta'].apply(lemmatize_words)

# Se une las listas de palabras de 'Pregunta' en strings
df['Pregunta'] = df['Pregunta'].apply(' '.join)

df['Pregunta']


# Se aplica el lematizador de Spacy para conservar solo las palabras con categorías gramaticales especificadas (sustantivos, adjetivos y verbos) y se devuelve el texto de la conversación con las palabras lematizadas.

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB"]):
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return texts_out


lemmatized_texts = lemmatization(df['Pregunta'])

def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return final

list_data_words = gen_words(lemmatized_texts)


# Acá ya se ve que las conversaciones de la columna 'Pregunta" no tienen acentos ni stopwords
df.head()


# #### Creación de Diccionario y Corpus
# En esta sección, se crea un diccionario de palabras (id2word) a partir de la lista de palabras tokenizadas (list_data_words). Luego, se crea un corpus en forma de representación de  bolsa de palabras utilizando el diccionario. Esto es necesario para el modelado de tópicos. 
# 
# ##### bag-of-words (BoW)
# Al utilizar la técnica de bag-of-words (BoW) convertimos la conversación en su vector equivalente de números. 
# Se crea una tupla (ID de palabra, frecuencia de aparición) para cada palabra. La variable new_tuple almacena estas tuplas.


# Se crea un diccionario de palabras a partir de la lista de palabras procesadas
id2word = corpora.Dictionary(list_data_words)

# Se crea un corpus de tuplas (bag-of-words) a partir del diccionario y las palabras procesadas
tuple_corpus = []
for word in list_data_words:
    new_tuple = id2word.doc2bow(word)
    tuple_corpus.append(new_tuple)
    
print(tuple_corpus)    


# #### Entrenamiento de modelo con LDA (Latent Dirichlet Allocation)
# En esta sección se entrena un modelo de tópicos utilizando el algoritmo LDA de Gensim. Se especifican los parámetros del modelo, como el número de tópicos, el tamaño de lote, el número de pases, etc.

# Se entrena un modelo LDA utilizando Gensim con los parámetros especificados.

lda_model = gensim.models.ldamodel.LdaModel(corpus=tuple_corpus, id2word=id2word, num_topics=7, random_state=100, update_every=1, chunksize=200, passes=10, alpha="auto")

# Se imprimen los tópicos del modelo LDA.

topics = lda_model.print_topics(num_topics=7, num_words=2)  
for topic in topics:
    print(topic)


# #### Tópicos descubiertos por el modelo LDA durante el proceso de entrenamiento
# - La función **show_topics()** es específica de Gensim y proporciona información sobre los tópicos descubiertos durante el entrenamiento del modelo LDA
# - Se muestran los tópicos junto con las palabras más relevantes que contribuyen a cada tópico y sus respectivas ponderaciones
# - El número que representa al identificador de un tópico es único. Esta identificación es interna y se genera automáticamente por el modelo LDA durante el proceso de entrenamiento. No tiene relación directa con ningún tópico específico en el conjunto de datos original ni con ningún orden específico de tópicos

lda_model.show_topics()

# #### Tópicos descubiertos por el modelo LDA para cada conversación del corpus otorgado
# - Después de ejecutar las siguientes funciones se imprimen los resultados mostrando por cada conversación, su bag-of-words correspondiente y los tópicos relevandos, primero ordenados por número de tópico y luego ordenados por probabilidad
# 

# Se crea una función para asignar tópicos a una conversación específica
def assign_topics_to_conversation(text):
    
    # Dentro de la función, se crea una representación de Bolsa de Palabras (BoW) para la conversación text utilizando el diccionario id2word. La función simple_preprocess se utiliza para preprocesar el texto, convirtiéndolo en minúsculas y eliminando acentos y caracteres no alfabéticos.
    bow = id2word.doc2bow(simple_preprocess(text, deacc=True))
    
    print("CONVERSACION: ")
    print(text)
    print("Bag Of Words(BOW) DE LA CONVERSACION: ")
    print(bow)
    topic_distribution = lda_model.get_document_topics(bow)
    print("TOPIC_DISTRIBUTION: ")
    print(topic_distribution)
    
    # Se ordenan los tópicos por probabilidad y se obtiene el tópico más probable
    
    def sort_by_second_element(x):
        return x[1]

    topic_distribution.sort(key=sort_by_second_element, reverse=True)

    print("TOPIC_DISTRIBUTION ORDENADO x PROBABILIDAD: ")
    print(topic_distribution)
    print("_______________________________________")
    # Devolver el número de tópico más probable
    return topic_distribution[0][0]

# Se aplica esta función a la columna 'Pregunta' del DataFrame y se crea una nueva columna llamada 'Topic' en el DataFrame para almacenar los tópicos asignados.
df['Topic'] = df['Pregunta'].apply(assign_topics_to_conversation)
df


# ### DataFrame Final
# La siguiente función crea una nueva columna en el DataFrame que figurará bajo el nombre de 'Topic Words'. 
# - 'Topic Words': Contiene las palabras clave más importantes del tópico correspondiente a la conversación de esa fila
# - 'Topic': Contiene el ID del tópico con la ponderación más alta para cada conversación

def show_topic_words(topic_id):
    topic_tuple_list = lda_model.show_topic(topic_id, topn=10)
    topic_words = [word for word, _ in topic_tuple_list]
    return " ".join(topic_words)

df['Topic Words'] = df['Topic'].apply(show_topic_words)
df