---
layout: model
title: Detect Chemicals in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_bc4chemd_chemicals
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

The automatic extraction of chemical information from text requires the recognition of chemical entity mentions as one of its key steps. 

This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP. The model detects chemical entities from a medical text.

## Predicted Entities

`CHEM`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc4chemd_chemicals_en_4.0.0_3.0_1658751849323.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc4chemd_chemicals_en_4.0.0_3.0_1658751849323.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_bc4chemd_chemicals", "en", "clinical/models")\
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

data = spark.createDataFrame([["""The main isolated compounds were triterpenes (alpha - amyrin, beta - amyrin, lupeol, betulin, betulinic acid, uvaol, erythrodiol and oleanolic acid) and phenolic acid derivatives from 4 - hydroxybenzoic acid (gallic and protocatechuic acids and isocorilagin)."""]]).toDF("text")

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

val ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_bc4chemd_chemicals", "en", "clinical/models")
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

val data = Seq("""The main isolated compounds were triterpenes (alpha - amyrin, beta - amyrin, lupeol, betulin, betulinic acid, uvaol, erythrodiol and oleanolic acid) and phenolic acid derivatives from 4 - hydroxybenzoic acid (gallic and protocatechuic acids and isocorilagin).""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------------+-----+
|ner_chunk                      |label|
+-------------------------------+-----+
|triterpenes                    |CHEM |
|alpha - amyrin                 |CHEM |
|beta - amyrin                  |CHEM |
|lupeol                         |CHEM |
|betulin                        |CHEM |
|betulinic acid                 |CHEM |
|uvaol                          |CHEM |
|erythrodiol                    |CHEM |
|oleanolic acid                 |CHEM |
|phenolic acid                  |CHEM |
|4 - hydroxybenzoic acid        |CHEM |
|gallic and protocatechuic acids|CHEM |
|isocorilagin                   |CHEM |
+-------------------------------+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bc4chemd_chemicals|
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

[https://github.com/cambridgeltl/MTL-Bioinformatics-2016](https://github.com/cambridgeltl/MTL-Bioinformatics-2016)

## Benchmarking

```bash
 label         precision  recall  f1-score  support 
 B-CHEM        0.7642     0.9536  0.8485    25346   
 I-CHEM        0.9446     0.9502  0.9474    29642   
 micro-avg     0.8517     0.9518  0.8990    54988   
 macro-avg     0.8544     0.9519  0.8979    54988   
 weighted-avg  0.8614     0.9518  0.9018    54988
```
