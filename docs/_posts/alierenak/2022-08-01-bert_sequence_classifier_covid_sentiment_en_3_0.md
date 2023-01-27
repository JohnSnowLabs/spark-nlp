---
layout: model
title: COVID-19 Sentiment Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_covid_sentiment
date: 2022-08-01
tags: [public_health, covid19_sentiment, en, licenced]
task: Sentiment Analysis
language: en
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
recommended: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [BioBERT](https://nlp.johnsnowlabs.com/2022/07/18/biobert_pubmed_base_cased_v1.2_en_3_0.html) based sentiment analysis model that can extract information from COVID-19 pandemic-related tweets. The model predicts whether a tweet contains positive, negative, or neutral sentiments about COVID-19 pandemic.

## Predicted Entities

`neutral`, `positive`, `negative`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_covid_sentiment_en_4.0.2_3.0_1659344524584.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_covid_sentiment_en_4.0.2_3.0_1659344524584.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_covid_sentiment", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
    ["British Department of Health confirms first two cases of in UK"],
    ["so my trip to visit my australian exchange student just got canceled bc of coronavirus. im heartbroken :("], 
    [ "I wish everyone to be safe at home and stop pandemic"]]
).toDF("text")

result = pipeline.fit(data).transform(data)
result.select("text", "class.result").show(truncate=False)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_covid_sentiment", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("British Department of Health confirms first two cases of in UK", 
                     "so my trip to visit my australian exchange student just got canceled bc of coronavirus. im heartbroken :(", 
                     "I wish everyone to be safe at home and stop pandemic"
)).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------------------------+----------+
|text                                                                                                     |result    |
+---------------------------------------------------------------------------------------------------------+----------+
|British Department of Health confirms first two cases of in UK                                           |[neutral] |
|so my trip to visit my australian exchange student just got canceled bc of coronavirus. im heartbroken :(|[negative]|
|I wish everyone to be safe at home and stop pandemic                                                     |[positive]|
+---------------------------------------------------------------------------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_covid_sentiment|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

Curated from several academic and in-house datasets.

## Benchmarking

```bash
       label  precision    recall  f1-score   support
    negative       0.96      0.97      0.97      3284
    positive       0.94      0.96      0.95      1207
     neutral       0.96      0.94      0.95      3232
    accuracy          -         -      0.96      7723
   macro-avg       0.95      0.96      0.96      7723
weighted-avg       0.96      0.96      0.96      7723
```
