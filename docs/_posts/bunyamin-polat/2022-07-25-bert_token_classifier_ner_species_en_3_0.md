---
layout: model
title: Detect Organism in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_species
date: 2022-07-25
tags: [en, ner, clinical, licensed, bertfortokenclassification]
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

Species-800 is a corpus for species entities, which is based on manually annotated abstracts. It comprises 800 PubMed abstracts that contain identified organism mentions.

This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.

## Predicted Entities

`SPECIES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_species_en_4.0.0_3.0_1658758056681.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_species", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("ner")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    ner_model,
    ner_converter   
    ])

data = spark.createDataFrame([["""As determined by 16S rRNA gene sequence analysis, strain 6C (T) represents a distinct species belonging to the class Betaproteobacteria and is most closely related to Thiomonas intermedia DSM 18155 (T) and Thiomonas perometabolis DSM 18570 (T) ."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_species", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("ner")
    .setCaseSensitive(True)
    .setMaxSentenceLength(512)

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, 
                                                   sentence_detector,
                                                   tokenizer,
                                                   ner_model,
                                                   ner_converter))

val data = Seq("""As determined by 16S rRNA gene sequence analysis, strain 6C (T) represents a distinct species belonging to the class Betaproteobacteria and is most closely related to Thiomonas intermedia DSM 18155 (T) and Thiomonas perometabolis DSM 18570 (T) .""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------+-------+
|ner_chunk              |label  |
+-----------------------+-------+
|6C (T)                 |SPECIES|
|Betaproteobacteria     |SPECIES|
|Thiomonas intermedia   |SPECIES|
|DSM 18155 (T)          |SPECIES|
|Thiomonas perometabolis|SPECIES|
|DSM 18570 (T)          |SPECIES|
+-----------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_species|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[https://species.jensenlab.org/](https://species.jensenlab.org/)

## Benchmarking

```bash
 label         precision  recall  f1-score  support
 B-SPECIES     0.6073     0.9374  0.7371    767     
 I-SPECIES     0.7418     0.8648  0.7986    1043    
 micro-avg     0.6754     0.8956  0.7701    1810    
 macro-avg     0.6745     0.9011  0.7678    1810    
 weighted-avg  0.6848     0.8956  0.7725    1810 
```
