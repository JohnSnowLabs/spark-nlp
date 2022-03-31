---
layout: model
title: Mapping Drugs With Their Corresponding Actions And Treatments
author: John Snow Labs
name: drug_action_treatment_mapper
date: 2022-03-31
tags: [en, chunkmapper, licensed, drug, action, treatment]
task: Entity Resolution
language: en
edition: Spark NLP for Healthcare 3.4.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps drugs with their corresponding `action` and `treatment`. `action` refers to the function of the drug, `treatment` refers to which disease the drug is used to treat.

## Predicted Entities

`action`, `treatment`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/drug_action_treatment_mapper_en_3.4.2_3.0_1648718401322.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

ner =  MedicalBertForTokenClassifier.pretrained("bert_token_classifier_drug_development_trials", "en", "clinical/models")\
      .setInputCols("token","sentence")\
      .setOutputCol("ner")

nerconverter = NerConverterInternal()\
      .setInputCols("sentence", "token", "ner")\
      .setOutputCol("ner_chunk")

chunkerMapper = ChunkMapperModel.pretrained("drug_action_treatment_mapper", "en", "clinical/models") \
      .setInputCols("ner_chunk")\
      .setOutputCol("relations")\
      .setRel("action") #or treatment

pipeline = Pipeline().setStages([document_assembler,
                                 sentence_detector,
                                 tokenizer, 
                                 ner, 
                                 nerconverter, 
                                 chunkerMapper])

text = ["""
The patient was given Aspagin, Warfarina Lusa
"""]

test_data = spark.createDataFrame([text]).toDF("text")

res = pipeline.fit(test_data).transform(test_data)

```
```scala
val document_assembler = DocumentAssembler()
         .setInputCol('text')
         .setOutputCol('document')

val sentence_detector = SentenceDetector()
         .setInputCols("document")
         .setOutputCol("sentence")

val tokenizer = Tokenizer()
         .setInputCols("sentence")
         .setOutputCol("token")

val ner =  MedicalBertForTokenClassifier.pretrained("bert_token_classifier_drug_development_trials", "en", "clinical/models")
         .setInputCols("token","sentence")
         .setOutputCol("ner")

val nerconverter = NerConverterInternal()
         .setInputCols(Array("sentence", "token", "ner"))
         .setOutputCol("ner_chunk")

val chunkerMapper = ChunkMapperModel.pretrained("drug_action_treatment_mapper", "en", "clinical/models") 
         .setInputCols("ner_chunk")
         .setOutputCol("relations")
         .setRel("action") 

val pipeline =  new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, ner, nerconverter, chunkerMapper ))


val text_data = Seq("The patient was given Aspagin, Warfarina Lusa").toDF("text")


val res = pipeline.fit(test_data).transform(test_data)

```
</div>

## Results

```bash
+--------------+--------------+-------------------------------+
|ner_chunk     |mapping_result|all_relations                  |
+--------------+--------------+-------------------------------+
|Aspagin       |Analgesic     |Anti-Inflammatory:::Antipyretic|
|Warfarina Lusa|Anticoagulant |                               |
+--------------+--------------+-------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|drug_action_treatment_mapper|
|Compatibility:|Spark NLP for Healthcare 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|8.7 MB|