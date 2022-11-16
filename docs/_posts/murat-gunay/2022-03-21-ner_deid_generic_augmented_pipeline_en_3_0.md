---
layout: model
title: Pipeline to Detect PHI for Deidentification (Generic - Augmented)
author: John Snow Labs
name: ner_deid_generic_augmented_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, deidentification, generic, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_deid_generic_augmented](https://nlp.johnsnowlabs.com/2021/06/30/ner_deid_generic_augmented_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_augmented_pipeline_en_3.4.1_3.0_1647869128382.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_deid_generic_augmented_pipeline", "en", "clinical/models")

pipeline.annotate("A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227.")
```
```scala
val pipeline = new PretrainedPipeline("ner_deid_generic_augmented_pipeline", "en", "clinical/models")

pipeline.annotate("A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227.")
```
</div>

## Results

```bash
+-------------------------------------------------+---------+
|chunk                                            |ner_label|
+-------------------------------------------------+---------+
|2093-01-13                                       |DATE     |
|David Hale                                       |NAME     |
|Hendrickson                                      |NAME     |
|Ora MR.                                          |LOCATION |
|7194334                                          |ID       |
|01/13/93                                         |DATE     |
|Oliveira                                         |NAME     |
|25                                               |AGE      |
|1-11-2000                                        |DATE     |
|Cocke County Baptist Hospital. 0295 Keats Street.|LOCATION |
|(302) 786-5227                                   |CONTACT  |
+-------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_generic_augmented_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
