---
layout: model
title: Vaccine Sentiment Classifier (BioBERT)
author: John Snow Labs
name: bert_sequence_classifier_vaccine_sentiment
date: 2022-07-28
tags: [public_health, vaccine_sentiment, en, licensed, sequence_classification]
task: Sentiment Analysis
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
recommended: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [BioBERT](https://nlp.johnsnowlabs.com/2022/07/18/biobert_pubmed_base_cased_v1.2_en_3_0.html) based sentimental analysis model that can extract information from COVID-19 Vaccine-related tweets. The model predicts whether a tweet contains positive, negative, or neutral sentiments about COVID-19 Vaccines.

## Predicted Entities

`neutral`, `positive`, `negative`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_VACCINE_STATUS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_vaccine_sentiment_en_4.0.0_3.0_1658995472179.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_vaccine_sentiment_en_4.0.0_3.0_1658995472179.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_vaccine_sentiment", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

text_list = ['A little bright light for an otherwise dark week. Thanks researchers, and frontline workers. Onwards.', 
             'People with a history of severe allergic reaction to any component of the vaccine should not take.', 
             '43 million doses of vaccines administrated worldwide...Production capacity of CHINA to reach 4 b']

data = spark.createDataFrame(text_list, StringType()).toDF("text")
result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_vaccine_sentiment", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("A little bright light for an otherwise dark week. Thanks researchers, and frontline workers. Onwards.", 
                     "People with a history of severe allergic reaction to any component of the vaccine should not take.", 
                     "43 million doses of vaccines administrated worldwide...Production capacity of CHINA to reach 4 b")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------------------------------------------------------------------------------+----------+
|text                                                                                                 |class     |
+-----------------------------------------------------------------------------------------------------+----------+
|A little bright light for an otherwise dark week. Thanks researchers, and frontline workers. Onwards.|[positive]|
|People with a history of severe allergic reaction to any component of the vaccine should not take.   |[negative]|
|43 million doses of vaccines administrated worldwide...Production capacity of CHINA to reach 4 b     |[neutral] |
+-----------------------------------------------------------------------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_vaccine_sentiment|
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

Curated from several academic and in-house datasets.

## Benchmarking

```bash
       label  precision    recall  f1-score   support 
     neutral       0.82      0.78      0.80      1007 
    positive       0.88      0.90      0.89      1002 
    negative       0.83      0.86      0.84       881 
    accuracy       -         -         0.85      2890 
   macro-avg       0.85      0.85      0.85      2890 
weighted-avg       0.85      0.85      0.85      2890 
```
