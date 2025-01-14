# Análisis de Sentimientos en Amazon Reviews 2023

## Descripción del Proyecto
Este proyecto aborda el análisis de sentimientos en las reseñas de productos de Amazon utilizando técnicas avanzadas de procesamiento de lenguaje natural (NLP) y algoritmos de machine learning. El objetivo principal es extraer patrones emocionales de las reseñas y analizar su relación con el comportamiento del consumidor.

La metodología aplicada incluye la limpieza y preprocesamiento del texto, la implementación de análisis de sentimientos mediante el modelo VADER, el balanceo de clases con SMOTE y la evaluación de múltiples modelos de machine learning, destacando la regresión logística como el enfoque más eficiente y preciso.

## Estructura del Código
El proyecto está dividido en varias etapas:

## Preprocesamiento de Datos
Limpieza de texto: eliminación de stopwords, lematización y tokenización.
Transformación de los datos para adaptarlos al análisis.

## Análisis de Sentimientos con VADER
Clasificación inicial de sentimientos en POSITIVE, NEGATIVE y NEUTRAL.
Visualización de la distribución de sentimientos antes y después del balanceo.

## Balanceo de Clases con SMOTE
Aplicación de SMOTE para abordar el desbalanceo de clases.
Visualización de la nueva distribución balanceada.

## Modelado con Machine Learning
Evaluación de varios modelos (Linear SVC, Multinomial Naive Bayes, Random Forest y Logistic Regression).
Optimización de hiperparámetros para Logistic Regression.

## Resultados de Regresión Logística
La regresión logística destacó como el modelo más equilibrado entre precisión y eficiencia computacional.
Métricas clave: precisión, recall, F1-score, curva ROC y matriz de confusión.

## Pruebas con Nuevas Reseñas
Se probaron reseñas adicionales utilizando el modelo entrenado, demostrando su capacidad para clasificar con precisión.
Resultados Obtenidos

## Métricas de Evaluación del Modelo
Accuracy: 92%
F1-Score (Macro): 86%
Curva ROC AUC:
Clase NEGATIVE: 0.96
Clase NEUTRAL: 0.97
Clase POSITIVE: 0.98

## Matriz de Confusión
Clase NEGATIVE: 12,443 correctamente clasificados, 1,743 mal clasificados.
Clase NEUTRAL: 14,269 correctamente clasificados, 1,693 mal clasificados.
Clase POSITIVE: 101,126 correctamente clasificados, 1,653 mal clasificados.

## Predicción de Nuevas Reseñas
Review: 'The product was fantastic! Exceeded my expectations.' --> Sentimiento: POSITIVE
Review: 'This is the worst product I’ve ever purchased. It broke immediately.' --> Sentimiento: NEGATIVE
Review: 'Amazing quality, very durable and worth the price.' --> Sentimiento: POSITIVE
Review: 'Completely useless, it didn’t work as described.' --> Sentimiento: NEGATIVE
Review: 'The product is great' --> Sentimiento: POSITIVE
Review: 'The quality is average' --> Sentimiento: NEUTRAL
Review: 'It works as described, but I wouldn't buy it again.' --> Sentimiento: NEUTRAL

## Archivos Incluidos
Código Principal (Amazon_Reviews_Sentiment_Analysis_compartido.ipynb)
Contiene las funciones para preprocesamiento, análisis de sentimientos y modelado.

## Modelo Entrenado (best_model_and_vectorizer.pkl)
Archivo pickle que almacena el modelo de regresión logística y el vectorizador TF-IDF.

## Resultados de Evaluación (results/)
Gráficos de curva ROC, matriz de confusión y distribución de sentimientos.


## Dataset (Amazon_Reviews_2023.csv)
Subconjunto de las reseñas de Amazon utilizado para este análisis.

## Requisitos del Sistema
Entorno Python
Python 3.8 o superior.
Librerías requeridas: scikit-learn, nltk, pandas, matplotlib, seaborn.

Instalar dependencias usando:
pip install -r requirements.txt
Requisitos Computacionales

Mínimo 16 GB de RAM.
Tiempo de ejecución estimado: ~10 minutos para el modelo Logistic Regression.

## Conclusiones
Logistic Regression es el modelo más eficiente y equilibrado, alcanzando altos niveles de precisión y F1-score con un tiempo de entrenamiento reducido.
La implementación de SMOTE fue crucial para abordar el problema de desbalanceo de clases, mejorando la clasificación de reseñas NEUTRAL y NEGATIVE.
Este análisis ofrece una herramienta práctica para empresas como Amazon para identificar productos problemáticos y personalizar recomendaciones basadas en sentimientos.

## Instrucciones para Uso

import pickle

# Cargar modelo y vectorizador
with open('best_model_and_vectorizer.pkl', 'rb') as f:
    data = pickle.load()

model = data['model']
vectorizer = data['vectorizer']

## Probar Nuevas Reseñas
new_reviews = ["Amazing product!", "Terrible experience, would not recommend."]
new_reviews_vectorized = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_vectorized)
print(predictions)

## Evaluar el Modelo
Visualizar la matriz de confusión y curva ROC con los scripts incluidos en Amazon_Reviews_Sentiment_Analysis_compartido.ipynb.
