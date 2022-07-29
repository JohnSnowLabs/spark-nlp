---
layout: model
title: Detect Disease Mentions (MedicalBertForTokenClassification) (BERT)
author: John Snow Labs
name: bert_token_classifier_disease_mentions_tweet
date: 2022-07-28
tags: [es, clinical, licensed, public_health, ner, token_classification, disease, tweet]
task: Named Entity Recognition
language: es
edition: Spark NLP for Healthcare 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is intended for detecting disease mentions in Spanish tweets and trained using the BertForTokenClassification method from the transformers library and [BERT based](https://huggingface.co/amine/bert-base-5lang-cased) embeddings.

## Predicted Entities

`ENFERMEDAD`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_disease_mentions_tweet_es_4.0.0_3.0_1659033666412.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols("sentence")\
  .setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_disease_mentions_tweet", "es", "clinical/models")\
  .setInputCols("token", "sentence")\
  .setOutputCol("label")\
  .setCaseSensitive(True)

ner_converter = NerConverter()\
  .setInputCols(["sentence","token","label"])\
  .setOutputCol("ner_chunk")


pipeline =  Pipeline(stages=[
                      documentAssembler,
                      sentenceDetector,
                      tokenizer,
                      tokenClassifier,
                      ner_converter])

model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

data = spark.createDataFrame(["El diagnóstico fueron varios. Principal: Neumonía en el pulmón derecho. Sinusitis de caballo, Faringitis aguda e infección de orina, también elevada. Gripe No. Estuvo hablando conmigo, sin exagerar, mas de media hora, dándome ánimo y fuerza y que sabe, porque ha visto"], StringType()).toDF("text")

result = model.transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_disease_mentions_tweet", "es", "clinical/models")
  .setInputCols(Array("token", "sentence"))
  .setOutputCol("label")
  .setCaseSensitive(True)

val ner_converter = new NerConverter()
  .setInputCols(Array("sentence","token","label"))
  .setOutputCol("ner_chunk")


val pipeline =  new Pipeline().setStages(Array(
                      documentAssembler,
                      sentenceDetector,
                      tokenizer,
                      tokenClassifier,
                      ner_converter))

val data = Seq(Array("El diagnóstico fueron varios. Principal: Neumonía en el pulmón derecho. Sinusitis de caballo, Faringitis aguda e infección de orina, también elevada. Gripe No. Estuvo hablando conmigo, sin exagerar, mas de media hora, dándome ánimo y fuerza y que sabe, porque ha visto"), StringType()).toDS().toDF("text")

val result = model.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------------+----------+
|chunk                |ner_label |
+---------------------+----------+
|Neumonía en el pulmón|ENFERMEDAD|
|Sinusitis            |ENFERMEDAD|
|Faringitis aguda     |ENFERMEDAD|
|infección de orina   |ENFERMEDAD|
|Gripe                |ENFERMEDAD|
+---------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_disease_mentions_tweet|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|461.7 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## Benchmarking

```bash
       label  precision    recall  f1-score   support
B-ENFERMEDAD       0.74      0.95      0.83      4243
I-ENFERMEDAD       0.64      0.79      0.71      1570
   micro avg       0.71      0.91      0.80      5813
   macro avg       0.69      0.87      0.77      5813
weighted avg       0.71      0.91      0.80      5813
```