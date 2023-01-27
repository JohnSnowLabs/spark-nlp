---
layout: model
title: Sentiment Analysis of German texts
author: John Snow Labs
name: classifierdl_bert_sentiment
date: 2021-09-09
tags: [de, sentiment, classification, open_source]
task: Sentiment Analysis
language: de
edition: Spark NLP 3.2.0
spark_version: 2.4
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model identifies the sentiments (positive or negative) in German texts.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_DE/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_De_SENTIMENT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_sentiment_de_3.2.0_2.4_1631184887201.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_bert_sentiment_de_3.2.0_2.4_1631184887201.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "de") \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")

fr_sentiment_pipeline = Pipeline(stages=[document, embeddings, sentimentClassifier])

light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result1 = light_pipeline.annotate("Spiel und Meisterschaft nicht spannend genug? Muss man jetzt den Videoschiedsrichter kontrollieren? Ich bin entsetzt...dachte der darf nur bei krassen Fehlentscheidungen ran. So macht der Fussball keinen Spass mehr.")

result2 = light_pipeline.annotate("Habe gestern am Mittwoch den #werder Podcast vermisst. Wie schnell man sich an etwas gewöhnt und darauf freut. Danke an @Plainsman74 für die guten Interviews und den Einblick hinter die Kulissen von @werderbremen. Angenehme Winterpause weiterhin!")
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

val sentimentClassifier = ClassifierDLModel.pretrained("classifierdl_bert_sentiment", "de") 
.setInputCols(Array("document", "sentence_embeddings")) 
.setOutputCol("class")

val fr_sentiment_pipeline = new Pipeline().setStages(Array(document, embeddings, sentimentClassifier))

val light_pipeline = LightPipeline(fr_sentiment_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

val result1 = light_pipeline.annotate("Spiel und Meisterschaft nicht spannend genug? Muss man jetzt den Videoschiedsrichter kontrollieren? Ich bin entsetzt...dachte der darf nur bei krassen Fehlentscheidungen ran. So macht der Fussball keinen Spass mehr.")

val result2 = light_pipeline.annotate("Habe gestern am Mittwoch den #werder Podcast vermisst. Wie schnell man sich an etwas gewöhnt und darauf freut. Danke an @Plainsman74 für die guten Interviews und den Einblick hinter die Kulissen von @werderbremen. Angenehme Winterpause weiterhin!")

```


{:.nlu-block}
```python
import nlu
nlu.load("de.classify.sentiment.bert").predict("""Habe gestern am Mittwoch den #werder Podcast vermisst. Wie schnell man sich an etwas gewöhnt und darauf freut. Danke an @Plainsman74 für die guten Interviews und den Einblick hinter die Kulissen von @werderbremen. Angenehme Winterpause weiterhin!""")
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
|Language:|de|

## Data Source

https://github.com/charlesmalafosse/open-dataset-for-sentiment-analysis/

## Benchmarking

```bash
label  precision    recall  f1-score   support
NEGATIVE       0.83      0.85      0.84       978
POSITIVE       0.94      0.93      0.94      2582
accuracy          -         -      0.91      3560
macro-avg       0.89      0.89      0.89      3560
weighted-avg       0.91      0.91      0.91      3560
```