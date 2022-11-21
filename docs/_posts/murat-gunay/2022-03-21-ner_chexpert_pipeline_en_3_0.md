---
layout: model
title: Pipeline to Detect Anatomical and Observation Entities in Chest Radiology Reports
author: John Snow Labs
name: ner_chexpert_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, chexpert, en]
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

This pretrained pipeline is built on the top of [ner_chexpert](https://nlp.johnsnowlabs.com/2021/09/30/ner_chexpert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chexpert_pipeline_en_3.4.1_3.0_1647867766035.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_chexpert_pipeline", "en", "clinical/models")


pipeline.annotate("FINAL REPORT HISTORY : Chest tube leak , to assess for pneumothorax . FINDINGS : In comparison with study of ___ , the endotracheal tube and Swan - Ganz catheter have been removed . The left chest tube remains in place and there is no evidence of pneumothorax. Mild atelectatic changes are seen at the left base.")
```
```scala
val pipeline = new PretrainedPipeline("ner_chexpert_pipeline", "en", "clinical/models")


pipeline.annotate("FINAL REPORT HISTORY : Chest tube leak , to assess for pneumothorax . FINDINGS : In comparison with study of ___ , the endotracheal tube and Swan - Ganz catheter have been removed . The left chest tube remains in place and there is no evidence of pneumothorax. Mild atelectatic changes are seen at the left base.")
```
</div>

## Results

```bash
|    | chunk                    | label   |
|---:|:-------------------------|:--------|
|  0 | endotracheal tube        | OBS     |
|  1 | Swan - Ganz catheter     | OBS     |
|  2 | left chest               | ANAT    |
|  3 | tube                     | OBS     |
|  4 | in place                 | OBS     |
|  5 | pneumothorax             | OBS     |
|  6 | Mild atelectatic changes | OBS     |
|  7 | left base                | ANAT    |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_chexpert_pipeline|
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
