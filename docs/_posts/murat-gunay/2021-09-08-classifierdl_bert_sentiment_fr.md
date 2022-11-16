---
layout: model
title: Sentiment Analysis of French texts
author: John Snow Labs
name: classifierdl_bert_sentiment
date: 2021-09-08
tags: [fr, sentiment, classification, open_source]
task: Sentiment Analysis
language: fr
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model identifies the sentiments (positive or negative) in French texts.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_FR/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_Fr_Sentiment.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_sentiment_fr_3.2.0_2.4_1631104713514.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

embeddings = BertSentenceEmbeddings\
.pretrained('labse', 'xx') \
.setInputCols(["document"])\
.setOutputCol("sentence_embeddings")

sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "fr") \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")

fr_sentiment_pipeline = Pipeline(stages=[document, embeddings, sentimentClassifier])

light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
result1 = light_pipeline.annotate("Mignolet vraiment dommage de ne jamais le voir comme titulaire")
result2 = light_pipeline.annotate("Je me sens bien, je suis heureux d'être de retour.")
print(result1["class"], result2["class"], sep = "\n")
```
```scala
val document = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val embeddings = BertSentenceEmbeddings
.pretrained("labse", "xx") 
.setInputCols(Array("document"))
.setOutputCol("sentence_embeddings")

val sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "fr") 
.setInputCols(Array("document", "sentence_embeddings")) 
.setOutputCol("class")

val fr_sentiment_pipeline = new Pipeline().setStages(Array(document, embeddings, sentimentClassifier))

val light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
val result1 = light_pipeline.annotate("Mignolet vraiment dommage de ne jamais le voir comme titulaire")
val result2 = light_pipeline.annotate("Je me sens bien, je suis heureux d'être de retour.")
```


{:.nlu-block}
```python
import nlu
nlu.load("fr.classify.sentiment.bert").predict("""Mignolet vraiment dommage de ne jamais le voir comme titulaire""")
```

</div>

## Results

```bash
['NEGATIVE']
['POSITIVE']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bert_sentiment|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|fr|

## Data Source

https://github.com/charlesmalafosse/open-dataset-for-sentiment-analysis/

## Benchmarking

```bash
precision    recall  f1-score   support

NEGATIVE       0.82      0.72      0.77       378
POSITIVE       0.92      0.95      0.94      1240

accuracy                           0.90      1618
macro avg       0.87      0.84      0.85      1618
weighted avg       0.90      0.90      0.90      1618
```
