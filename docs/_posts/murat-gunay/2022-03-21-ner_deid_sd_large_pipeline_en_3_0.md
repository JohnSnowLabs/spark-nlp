---
layout: model
title: Pipeline to Detect PHI in Text
author: John Snow Labs
name: ner_deid_sd_large_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, deidentification, en]
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

This pretrained pipeline is built on the top of [ner_deid_sd_large](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_sd_large_en.html) model.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_sd_large_pipeline_en_3.4.1_3.0_1647870104226.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_sd_large_pipeline_en_3.4.1_3.0_1647870104226.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_deid_sd_large_pipeline", "en", "clinical/models")

pipeline.annotate("""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25-year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.""")
```
```scala
val pipeline = new PretrainedPipeline("ner_deid_sd_large_pipeline", "en", "clinical/models")

pipeline.annotate("A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25-year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.")
```
</div>

## Results

```bash
+-----------------------------+--------+
|chunks                       |entities|
+-----------------------------+--------+
|2093-01-13                   |DATE    |
|David Hale                   |NAME    |
|Hendrickson, Ora             |NAME    |
|7194334                      |ID      |
|01/13/93                     |DATE    |
|Oliveira                     |NAME    |
|1-11-2000                    |DATE    |
|Cocke County Baptist Hospital|LOCATION|
|0295 Keats Street            |LOCATION|
|786-5227                     |CONTACT |
|Brothers Coal-Mine           |LOCATION|
+-----------------------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_sd_large_pipeline|
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
