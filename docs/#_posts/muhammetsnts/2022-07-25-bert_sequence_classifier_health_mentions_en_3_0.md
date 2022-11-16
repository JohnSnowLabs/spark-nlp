---
layout: model
title: Public Health Mention Sequence Classifier (PHS-BERT)
author: John Snow Labs
name: bert_sequence_classifier_health_mentions
date: 2022-07-25
tags: [public_health, en, licensed, sequence_classification, health, mention]
task: Text Classification
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [PHS-BERT](https://arxiv.org/abs/2204.04521) based sequence classification model that can classify public health mentions in social media text. Mentions are classified into three labels about personal health situation, figurative mention and other mentions. More detailed information about classes as follows:

`health_mention`: The text contains a health mention that specifically indicating someone's health situation.  This means someone has a certain disease or symptoms including death. e.g.; *My PCR test is positive. I have a severe joint pain, mucsle pain and headache right now.*

`other_mention`: The text contains a health mention; however does not states a spesific person's situation. General health mentions like informative mentions, discussion about disease etc. e.g.; *Aluminum is a light metal that causes dementia and Alzheimer's disease.*

`figurative_mention`: The text mention specific disease or symptom but it is used metaphorically, does not contain health-related information. e.g.; *I don't wanna fall in love. If I ever did that, I think I'd have a heart attack.*

## Predicted Entities

`figurative_mention`, `other_mention`, `health_mention`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_health_mentions_en_4.0.0_3.0_1658746315237.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
# Sample Python Code

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([["Another uncle of mine had a heart attack and passed away. Will be cremated Saturday I think I ve gone numb again RIP Uncle Mike"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq("Another uncle of mine had a heart attack and passed away. Will be cremated Saturday I think I ve gone numb again RIP Uncle Mike")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------------------------------------------------------------------------------------------------------------+----------------+
|text                                                                                                                           |class           |
+-------------------------------------------------------------------------------------------------------------------------------+----------------+
|Another uncle of mine had a heart attack and passed away. Will be cremated Saturday I think I ve gone numb again RIP Uncle Mike|[health_mention]|
+-------------------------------------------------------------------------------------------------------------------------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_health_mentions|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

Curated from several academic and in-house datasets.

## Benchmarking

```bash
             label  precision    recall  f1-score   support 
    health_mention       0.85      0.86      0.86      1352 
     other_mention       0.90      0.89      0.89      2151 
figurative_mention       0.86      0.87      0.86      1386 
          accuracy       -         -         0.87      4889 
         macro-avg       0.87      0.87      0.87      4889 
      weighted-avg       0.87      0.87      0.87      4889 
```
