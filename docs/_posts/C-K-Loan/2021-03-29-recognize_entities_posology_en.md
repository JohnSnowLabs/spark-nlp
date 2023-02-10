---
layout: model
title: Recognize Posology Pipeline
author: John Snow Labs
name: recognize_entities_posology
date: 2021-03-29
tags: [ner, named_entity_recognition, pos, parts_of_speech, posology, ner_posology, pipeline, en, licensed]
task: [Named Entity Recognition, Part of Speech Tagging]
language: en
edition: Spark NLP for Healthcare 3.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline detects drugs, dosage, form, frequency, duration, route, and drug strength in text.

## Predicted Entities
`DRUG`, `STRENGTH`, `DURATION`, `FREQUENCY`, `FORM`, `DOSAGE`, `ROUTE`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/recognize_entities_posology_en_3.0.0_3.0_1617042229126.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/recognize_entities_posology_en_3.0.0_3.0_1617042229126.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline('recognize_entities_posology', 'en', 'clinical/models')

annotations =  pipeline.fullAnnotate("""The patient was perscriped 50MG penicilin for is headache""")[0]

annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("recognize_entities_posology", "en", "clinical/models")
val result = pipeline.fullAnnotate("""The patient was perscriped 50MG penicilin for is headache""")(0)

```

{:.nlu-block}
```python
import nlu

result_df = nlu.load('ner.posology').predict("""The patient was perscriped 50MG penicilin for is headache""")
result_df

```
</div>

## Results

```bash
+-----------------------------------------+
|result                                   |
+-----------------------------------------+
|[O, O, O, O, B-Strength, B-Drug, O, O, O]|
+-----------------------------------------+

+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|ner                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[[named_entity, 0, 2, O, [word -> The, confidence -> 1.0], []], [named_entity, 4, 10, O, [word -> patient, confidence -> 0.9993], []], [named_entity, 12, 14, O, [word -> was, confidence -> 1.0], []], [named_entity, 16, 25, O, [word -> perscriped, confidence -> 0.9985], []], [named_entity, 27, 30, B-Strength, [word -> 50MG, confidence -> 0.9966], []], [named_entity, 32, 40, B-Drug, [word -> penicilin, confidence -> 0.9934], []], [named_entity, 42, 44, O, [word -> for, confidence -> 0.9999], []], [named_entity, 46, 47, O, [word -> is, confidence -> 0.9468], []], [named_entity, 49, 56, O, [word -> headache, confidence -> 0.9805], []]]|
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|recognize_entities_posology|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter
