---
layout: model
title: Drug Reviews Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_drug_reviews_webmd
date: 2022-07-28
tags: [en, clinical, licensed, public_health, classifier, sequence_classification, drug, review]
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

This model is a [BioBERT based](https://github.com/dmis-lab/biobert) classifier that can classify drug reviews from WebMD.com

## Predicted Entities

`negative`, `positive`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_CHANGE_DRUG_TREATMENT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_drug_reviews_webmd_en_4.0.0_3.0_1659008484818.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_drug_reviews_webmd", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])


data = spark.createDataFrame(["While it has worked for me, the sweating and chills especially at night when trying to sleep are very off putting and I am not sure if I will stick with it very much longer. My eyese no longer feel like there is something in them and my mouth is definitely not as dry as before but the side effects are too invasive for my liking.",
                              "I previously used Cheratussin but was now dispensed Guaifenesin AC as a cheaper alternative. This stuff does n t work as good as Cheratussin and taste like cherry flavored sugar water."], StringType()).toDF("text")
                               
result = pipeline.fit(data).transform(data)

result.select("text", "class.result").show(truncate=False)
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_drug_reviews_webmd", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val data = Seq(Array("While it has worked for me, the sweating and chills especially at night when trying to sleep are very off putting and I am not sure if I will stick with it very much longer. My eyese no longer feel like there is something in them and my mouth is definitely not as dry as before but the side effects are too invasive for my liking.",
                     "I previously used Cheratussin but was now dispensed Guaifenesin AC as a cheaper alternative. This stuff does n t work as good as Cheratussin and taste like cherry flavored sugar water.")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------+
|text                                                                                                                                                                                                                                                                                                                                      |result    |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------+
|While it has worked for me, the sweating and chills especially at night when trying to sleep are very off putting and I am not sure if I will stick with it very much longer. My eyese no longer feel like there is something in them and my mouth is definitely not as dry as before but the side effects are too invasive for my liking.|[negative]|
|I previously used Cheratussin but was now dispensed Guaifenesin AC as a cheaper alternative. This stuff does n t work as good as Cheratussin and taste like cherry flavored sugar water .                                                                                                                                                 |[positive]|
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_drug_reviews_webmd|
|Compatibility:|Healthcare NLP 4.0.0+|
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
    negative     0.8589    0.8234    0.8408      1042
    positive     0.8612    0.8901    0.8754      1283
    accuracy     -         -         0.8602      2325
   macro-avg     0.8600    0.8568    0.8581      2325
weighted-avg     0.8602    0.8602    0.8599      2325
```