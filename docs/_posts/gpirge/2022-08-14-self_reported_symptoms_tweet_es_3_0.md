---
layout: model
title: Self-Reported Covid-19 Symptoms Classifier
author: John Snow Labs
name: self_reported_symptoms_tweet
date: 2022-08-14
tags: [es, clinical, licensed, public_health, classifier, covid_19, tweet, symptom]
task: Text Classification
language: es
edition: Spark NLP for Healthcare 4.0.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a classification model, trained by using the ClassifierDLApproach annotator and can classify the origin of symptoms related to Covid-19 from Spanish tweets. 
This model is intended for direct use as a classification model and the target classes are: Lit-News_mentions, Self_reports, non-personal_reports.

## Predicted Entities

`Lit-News_mentions`, `Self_reports`, `non-personal_reports`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/self_reported_symptoms_tweet_es_4.0.2_3.0_1660447906015.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
	.setInputCols(["document"])\
	.setOutputCol("sentence")

tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_scielo_300d","es","clinical/models")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("embeddings")
 
sentence_embeddings = SentenceEmbeddings() \
   .setInputCols(["document", "embeddings"]) \
   .setOutputCol("sentence_embeddings") \
   .setPoolingStrategy("AVERAGE")

classifier = ClassifierDLModel.pretrained('self_reported_symptoms_tweet') \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("class")    

pipeline = Pipeline(stages=[
    document,
    sentenceDetectorDL, 
    tokenizer,
    word_embeddings,
    sentence_embeddings,
    classifier
])

data = spark.createDataFrame(["Las vacunas 3 y hablamos inminidad vivo      Son bichito vivo dentro de lÃ­quido de la vacuna suelen tener reacciones alÃ©rgicas si que sepan o   Covid19 es un bichito de un resfriado comÃºn es normal 48 horas tenga poco de fiebre y con cuerpo raro ",
                              "Yo pense que me estaba dando el  coronavirus porque cuando me levante  casi no podia respirar pero que si era que tenia la nariz topada de mocos.",
                              "Tos, dolor de garganta y fiebre, los síntomas más reportados por los porteños con coronavirus"], StringType()).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
    .setInputCols(Array("document"))
	  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_scielo_300d","es","clinical/models")
	  .setInputCols(Array("sentence","token"))
	  .setOutputCol("embeddings")
 
val sentence_embeddings = new SentenceEmbeddings()
    .setInputCols(Array("document", "embeddings"))
    .setOutputCol("sentence_embeddings")
    .setPoolingStrategy("AVERAGE")

val classifier = ClassifierDLModel.pretrained('self_reported_symptoms_tweet')
    .setInputCols("sentence_embeddings")
    .setOutputCol("class")

val pipeline = new PipelineModel().setStages(Array(document_assembler,
                                                   sentenceDetectorDL, 
                                                   tokenizer,
                                                   word_embeddings,
                                                   sentence_embeddings,
                                                   classifier
                                                   ))

val data = Seq(Array("Las vacunas 3 y hablamos inminidad vivo      Son bichito vivo dentro de lÃ­quido de la vacuna suelen tener reacciones alÃ©rgicas si que sepan o   Covid19 es un bichito de un resfriado comÃºn es normal 48 horas tenga poco de fiebre y con cuerpo raro ",
                              "Yo pense que me estaba dando el  coronavirus porque cuando me levante  casi no podia respirar pero que si era que tenia la nariz topada de mocos.",
                              "Tos, dolor de garganta y fiebre, los síntomas más reportados por los porteños con coronavirus")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+------------------------------------------------------------------------------------------------------------------------+----------------------+
|                                                                                                                    text|                result|
+------------------------------------------------------------------------------------------------------------------------+----------------------+
|Las vacunas 3 y hablamos inminidad vivo      Son bichito vivo dentro de lÃ­quido de la vacuna suelen tener reacciones...|[non-personal_reports]|
|Yo pense que me estaba dando el  coronavirus porque cuando me levante  casi no podia respirar pero que si era que ten...|        [Self_reports]|
|                           Tos, dolor de garganta y fiebre, los síntomas más reportados por los porteños con coronavirus|   [Lit-News_mentions]|
+------------------------------------------------------------------------------------------------------------------------+----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|self_reported_symptoms_tweet|
|Compatibility:|Spark NLP for Healthcare 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|es|
|Size:|21.9 MB|

## References

The dataset is Covid-19-specific and consists of tweets collected via a series of keywords associated with that disease.

## Benchmarking

```bash
               label  precision    recall  f1-score   support
   Lit-News_mentions       0.91      0.95      0.93       727
        Self_reports       0.66      0.76      0.71       216
non-personal_reports       0.72      0.56      0.63       305
            accuracy       -         -         0.82      1248
           macro-avg       0.76      0.76      0.76      1248
        weighted-avg       0.82      0.82      0.82      1248
```
