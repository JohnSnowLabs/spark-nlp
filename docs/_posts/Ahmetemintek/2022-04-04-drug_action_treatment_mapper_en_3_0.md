---
layout: model
title: Mapping Drugs With Their Corresponding Actions And Treatments
author: John Snow Labs
name: drug_action_treatment_mapper
date: 2022-04-04
tags: [en, chunkmapping, chunkmapper, drug, action, treatment, licensed]
task: Chunk Mapping
language: en
edition: Healthcare NLP 3.4.2
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps drugs with their corresponding `action` and `treatment`. `action` refers to the function of the drug in various body systems, `treatment` refers to which disease the drug is used to treat

## Predicted Entities

`action`, `treatment`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/drug_action_treatment_mapper_en_3.4.2_3.0_1649098201229.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .setInputCols(["token","sentence"])\
    .setOutputCol("ner")

nerconverter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("drug")

chunkerMapper = ChunkMapperModel.pretrained("drug_action_treatment_mapper", "en", "clinical/models") \
    .setInputCols("drug")\
    .setOutputCol("relations")\
    .setRel("treatment") #or action

pipeline = Pipeline().setStages([document_assembler,
                                sentence_detector,
                                tokenizer,
                                ner,
                                nerconverter,
                                chunkerMapper])

text = ["""The patient is a 71-year-old female patient of Dr. X. and she was given Aklis and Dermovate.
Cureent Medications: Diprivan, Proventil """]

data = spark.createDataFrame([text]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

val tokenizer = Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val ner =  MedicalBertForTokenClassifier.pretrained("bert_token_classifier_drug_development_trials", "en", "clinical/models")
    .setInputCols(Array("token","sentence"))
    .setOutputCol("ner")

val nerconverter = NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("drug")

val chunkerMapper = ChunkMapperModel.pretrained("drug_action_treatment_mapper", "en", "clinical/models")
    .setInputCols("drug")
    .setOutputCol("relations")
    .setRel("treatment")

val pipeline =  new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, ner, nerconverter, chunkerMapper ))

val text_data = Seq("""The patient is a 71-year-old female patient of Dr. X. and she was given Aklis and Dermovate.
Cureent Medications: Diprivan, Proventil""").toDF("text")

val res = pipeline.fit(test_data).transform(test_data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.drug_to_action_treatment").predict("""
The patient is a 71-year-old female patient of Dr. X. and she was given Aklis and Dermovate.
Cureent Medications: Diprivan, Proventil
""")
```

</div>

## Results

```bash
+---------+------------------+--------------------------------------------------------------+
|Drug     |Treats            |Pharmaceutical Action                                         |
+---------+------------------+--------------------------------------------------------------+
|Aklis    |Hyperlipidemia    |Hypertension:::Diabetic Kidney Disease:::Cerebrovascular...   |
|Dermovate|Lupus             |Discoid Lupus Erythematosus:::Empeines:::Psoriasis:::Eczema...|
|Diprivan |Infection         |Laryngitis:::Pneumonia:::Pharyngitis                          |
|Proventil|Addison's Disease |Allergic Conjunctivitis:::Anemia:::Ankylosing Spondylitis     |
+---------+------------------+--------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|drug_action_treatment_mapper|
|Compatibility:|Healthcare NLP 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|8.7 MB|
