---
layout: model
title: Classification of Vaccine Tweets
author: John Snow Labs
name: bert_sequence_classifier_self_reported_vaccine_status_tweet
date: 2022-07-27
tags: [bert_sequence_classifier, bert, en, licensed, vaccine]
task: Text Classification
language: en
edition: Spark NLP for Healthcare 3.5.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

[SMM4H 2022](https://healthlanguageprocessing.org/smm4h-2022/) - Task 6 - Classification of tweets indicating self-reported COVID-19 vaccination status. This task involves the identification of self-reported COVID-19 vaccination status in English tweets.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_self_reported_vaccine_status_tweet_en_3.5.0_3.0_1658928804893.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_vaccine_status_tweet", "en", "clinical/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

example = spark.createDataFrame([["I came to a point finally and i've vaccinated, didnt feel pain.Suggest everyone"]]).toDF("text")

result = pipeline.fit(example).transform(example)

result.select("class.result", "text").show(truncate=False)
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document")) 
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_self_reported_vaccine_status_tweet", "en", "clinical/models")
  .setInputCols(Array("document","token"))
  .setOutputCol("class")

val pipeline = new Pipeline.setStages(Array(document_assembler, tokenizer, sequenceClassifier))

# couple of simple examples
val example = Seq("I came to a point finally and i've vaccinated, didnt feel pain.Suggest everyone").toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+--------------------+-----------------+
|                text|           result|
+--------------------+-----------------+
|  Am  I my brothe...|   [Self_reports]|     
|  Covid day 4.   ...|   [Self_reports]|     
|  For a minute  I...|   [Self_reports]|     
|  Got dose 1 of t...|   [Self_reports]|     
|  It s been month...|   [Self_reports]|     
|  Jim   Larranaga...|   [Self_reports]|     
|  My young and he...|   [Self_reports]|     
|  Received my 1st...|   [Self_reports]|     
|  Science is amaz...|   [Self_reports]|     
| I got my 2nd  CO...|   [Self_reports]|     
| I just finished ...|   [Self_reports]|     
| I just found out...|   [Self_reports]|     
| I took the first...|   [Self_reports]|     
|      There is no...|[Vaccine_chatter]|     
|    Hello hypocri...|[Vaccine_chatter]|     
|  But...checked m...|   [Self_reports]|     
|  Is there a peti...|[Vaccine_chatter]|     
|  Maybe people do...|[Vaccine_chatter]|     
| I just seen this...|   [Self_reports]|     
| I m sorry if thi...|[Vaccine_chatter]|     
+--------------------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_self_reported_vaccine_status_tweet|
|Compatibility:|Spark NLP for Healthcare 3.5.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## Benchmarking

```bash
          label  precision    recall  f1-score   support
   Self_reports       0.79      0.78      0.78       311
Vaccine_chatter       0.97      0.97      0.97      2410
       accuracy        -         -        0.95      2721
      macro-avg       0.88      0.88      0.88      2721
   weighted-avg       0.95      0.95      0.95      2721
```