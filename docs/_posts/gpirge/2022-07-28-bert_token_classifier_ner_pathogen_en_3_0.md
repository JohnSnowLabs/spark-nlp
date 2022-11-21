---
layout: model
title: Detect Pathogen, Medical Condition and Medicine (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_pathogen
date: 2022-07-28
tags: [licensed, clinical, en, ner, pathogen, medical_condition, medicine, berfortokenclassification]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained named entity recognition (NER) model is a deep learning model for detecting medical conditions (influenza, headache, malaria, etc), medicine (aspirin, penicillin, methotrexate) and pathogens (Corona Virus, Zika Virus, E. Coli, etc) in clinical texts. This model is trained with [BertForTokenClassification] method from transformers library and imported into Spark NLP.

## Predicted Entities

`medicine`, `medical_condition`, `pathogen`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_pathogen_en_4.0.0_3.0_1659033806498.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

sentenceDetector = SentenceDetectorDLModel.pretrained()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(['sentence']) \
    .setOutputCol('token')

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_pathogen", "en", "clinical/models")\
    .setInputCols(['token', "sentence"])\
    .setOutputCol("label")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
    .setInputCols(["sentence","token","label"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler,
    sentenceDetector, 
    tokenizer,
    tokenClassifier,
    ner_converter
])


data = spark.createDataFrame([["""Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin. This can progress to loss of skin color, a fast heart rate as it becomes more severe; while it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions."""]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_pathogen", "en", "clinical/models")
    .setInputCols(Array("token", 'sentence'))
    .setOutputCol("label")
    .setCaseSensitive(True)

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence","token","label"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documenter, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val data = Seq(Array("Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin. This can progress to loss of skin color, a fast heart rate as it becomes more severe; while it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions.")).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------+----------------+
|ticker         |label           |
+---------------+----------------+
|Racecadotril   |Medicine        |
|loperamide     |Medicine        |
|Diarrhea       |MedicalCondition|
|loose          |MedicalCondition|
|liquid         |MedicalCondition|
|watery         |MedicalCondition|
|bowel movements|MedicalCondition|
|dehydration    |MedicalCondition|
|loss           |MedicalCondition|
|color          |MedicalCondition|
|fast           |MedicalCondition|
|heart rate     |MedicalCondition|
|rabies virus   |Pathogen        |
|Lyssavirus     |Pathogen        |
|Ephemerovirus  |Pathogen        |
+---------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_pathogen|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## Benchmarking

```bash
           label  precision    recall  f1-score   support
MedicalCondition       0.73      0.78      0.75        49
        Medicine       0.95      0.95      0.95        38
        Pathogen       0.77      0.91      0.83        11
       micro-avg       0.82      0.86      0.84        98
       macro-avg       0.82      0.88      0.84        98
    weighted-avg       0.82      0.86      0.84        98
```
