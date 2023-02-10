---
layout: model
title: Pipeline to Detect PHI for Deidentification (Sub Entity)
author: John Snow Labs
name: ner_deid_subentity_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, deid, de]
task: Named Entity Recognition
language: de
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_deid_subentity](https://nlp.johnsnowlabs.com/2022/01/06/ner_deid_subentity_de.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_pipeline_de_3.4.1_3.0_1647887751010.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_pipeline_de_3.4.1_3.0_1647887751010.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
pipeline = PretrainedPipeline("ner_deid_subentity_pipeline", "de", "clinical/models")


pipeline.annotate("Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.")
```
```scala
val pipeline = new PretrainedPipeline("ner_deid_subentity_pipeline", "de", "clinical/models")


pipeline.annotate("Michael Berger wird am Morgen des 12 Dezember 2018 ins St. Elisabeth-Krankenhaus in Bad Kissingen eingeliefert. Herr Berger ist 76 Jahre alt und hat zu viel Wasser in den Beinen.")
```
</div>

## Results

```bash
+-------------------------+-------------------------+
|chunk                    |ner_deid_subentity_chunk |
+-------------------------+-------------------------+
|Michael Berger           |PATIENT                  |
|12 Dezember 2018         |DATE                     |
|St. Elisabeth-Krankenhaus|HOSPITAL                 |
|Bad Kissingen            |CITY                     |
|Berger                   |PATIENT                  |
|76                       |AGE                      |
+-------------------------+-------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|de|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
