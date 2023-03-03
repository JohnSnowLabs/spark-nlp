---
layout: model
title: Pipeline to Detect PHI for Deidentification
author: John Snow Labs
name: bert_token_classifier_ner_deid_pipeline
date: 2022-03-21
tags: [licensed, ner, deid, en]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_deid](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_deid_en.html) model.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_deid_pipeline_en_3.4.1_3.0_1647863928244.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_deid_pipeline_en_3.4.1_3.0_1647863928244.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_deid_pipeline", "en", "clinical/models")

pipeline.annotate("A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.")
```
```scala
val pipeline = new PretrainedPipeline("bert_token_classifier_ner_deid_pipeline", "en", "clinical/models")

pipeline.annotate("A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.")
```
</div>

## Results

```bash
+-----------------------------+-------------+
|chunk                        |ner_label    |
+-----------------------------+-------------+
|2093-01-13                   |DATE         |
|David Hale                   |DOCTOR       |
|Hendrickson, Ora             |PATIENT      |
|7194334                      |MEDICALRECORD|
|Oliveira                     |PATIENT      |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street            |STREET       |
|302) 786-5227                |PHONE        |
|Brothers Coal-Mine           |ORGANIZATION |
+-----------------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_deid_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverter