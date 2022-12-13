---
layout: model
title: Self Report Age Classifier (BioBERT - Reddit)
author: John Snow Labs
name: bert_sequence_classifier_exact_age_reddit
date: 2022-07-26
tags: [licensed, clinical, en, classifier, sequence_classification, age, public_health]
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

This model is a [BioBERT based](https://github.com/dmis-lab/biobert) classifier that can classify self-report the exact age into social media forum (Reddit) posts.

## Predicted Entities

`self_report_age`, `no_report`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_AGE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_exact_age_reddit_en_4.0.0_3.0_1658852929276.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_exact_age_reddit_en_4.0.0_3.0_1658852929276.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_exact_age_reddit", "en", "clinical/models")\
    .setInputCols(["document",'token'])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame(["Is it bad for a 19 year old it's been getting worser.",
                              "I was about 10. So not quite as young as you but young."], StringType()).toDF("text")
                              
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

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_exact_age_reddit", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("Is it bad for a 19 year old it's been getting worser.",
                     "I was about 10. So not quite as young as you but young.")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------------------------------------+-----------------+
|text                                                   |result           |
+-------------------------------------------------------+-----------------+
|Is it bad for a 19 year old it's been getting worser.  |[self_report_age]|
|I was about 10. So not quite as young as you but young.|[no_report]      |
+-------------------------------------------------------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_exact_age_reddit|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

The dataset is disease-specific and consists of posts collected via a series of keywords associated with dry eye disease.

## Benchmarking

```bash
          label  precision    recall  f1-score   support
      no_report     0.9324    0.9577    0.9449      1325
self_report_age     0.9124    0.8637    0.8874       675
       accuracy     -         -         0.9260      2000
      macro-avg     0.9224    0.9107    0.9161      2000
   weighted-avg     0.9256    0.9260    0.9255      2000
```
