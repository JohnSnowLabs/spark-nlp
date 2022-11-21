---
layout: model
title: Detect Genes/Proteins (BC2GM) in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_bc2gm_gene
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

The BioCreative II Gene Mention Recognition (BC2GM) Dataset contains data where participants are asked to identify a gene mentioned in a sentence by giving its start and end characters. The training set consists of a set of sentences and a set of gene mentions (GENE annotations) in the English language for each sentence. 

This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP. The model detects genes/proteins from a medical text.

## Predicted Entities

`GENE/PROTEIN`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc2gm_gene_en_4.0.0_3.0_1658751400656.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_bc2gm_gene", "en", "clinical/models")\
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

data = spark.createDataFrame([["""ROCK-I, Kinectin, and mDia2 can bind the wild type forms of both RhoA and Cdc42 in a GTP-dependent manner in vitro. These results support the hypothesis that in the presence of tryptophan the ribosome translating tnaC blocks Rho ' s access to the boxA and rut sites, thereby preventing transcription termination."""]]).toDF("text")

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

val ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_bc2gm_gene", "en", "clinical/models")
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

val data = Seq("""ROCK-I, Kinectin, and mDia2 can bind the wild type forms of both RhoA and Cdc42 in a GTP-dependent manner in vitro. These results support the hypothesis that in the presence of tryptophan the ribosome translating tnaC blocks Rho ' s access to the boxA and rut sites, thereby preventing transcription termination.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------+------------+
|ner_chunk|label       |
+---------+------------+
|ROCK-I   |GENE/PROTEIN|
|Kinectin |GENE/PROTEIN|
|mDia2    |GENE/PROTEIN|
|RhoA     |GENE/PROTEIN|
|Cdc42    |GENE/PROTEIN|
|tnaC     |GENE/PROTEIN|
|Rho      |GENE/PROTEIN|
|boxA     |GENE/PROTEIN|
|rut sites|GENE/PROTEIN|
+---------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bc2gm_gene|
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
 label           precision  recall  f1-score  support 
 B-GENE/PROTEIN  0.7907     0.9320  0.8556    6325    
 I-GENE/PROTEIN  0.8350     0.8651  0.8498    8776    
 micro-avg       0.8151     0.8931  0.8523    15101   
 macro-avg       0.8129     0.8986  0.8527    15101   
 weighted-avg    0.8165     0.8931  0.8522    15101 
```
