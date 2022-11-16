---
layout: model
title: Detect Organism in Medical Texts
author: John Snow Labs
name: bert_token_classifier_ner_linnaeus_species
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

This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP. The model detects species entities in a biomedical text

## Predicted Entities

`SPECIES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_linnaeus_species_en_4.0.0_3.0_1658755473753.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_linnaeus_species", "en", "clinical/models")\
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

data = spark.createDataFrame([["""First identified in chicken, vigilin homologues have now been found in human (6), Xenopus laevis (7), Drosophila melanogaster (8) and Schizosaccharomyces pombe."""]]).toDF("text")

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

val ner_model = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_linnaeus_species", "en", "clinical/models")
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

val data = Seq("""First identified in chicken, vigilin homologues have now been found in human (6), Xenopus laevis (7), Drosophila melanogaster (8) and Schizosaccharomyces pombe.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------------------+-------+
|ner_chunk                |label  |
+-------------------------+-------+
|chicken                  |SPECIES|
|human                    |SPECIES|
|Xenopus laevis           |SPECIES|
|Drosophila melanogaster  |SPECIES|
|Schizosaccharomyces pombe|SPECIES|
+-------------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_linnaeus_species|
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
B-SPECIES     0.6391     0.9204  0.7544    1433    
I-SPECIES     0.8297     0.7071  0.7635    799     
micro-avg     0.6863     0.8441  0.7571    2232    
macro-avg     0.7344     0.8138  0.7589    2232    
weighted-avg  0.7073     0.8441  0.7576    2232 
```
