---
layout: model
title: Pipeline to Detect PHI in Text (enriched-biobert)
author: John Snow Labs
name: ner_deid_enriched_biobert_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, deidentification, enriched_biobert, en]
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

This pretrained pipeline is built on the top of [ner_deid_enriched_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_enriched_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_biobert_pipeline_en_3.4.1_3.0_1647868393082.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_deid_enriched_biobert_pipeline", "en", "clinical/models")

pipeline.annotate("""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25-year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.""")
```
```scala
val pipeline = new PretrainedPipeline("ner_deid_enriched_biobert_pipeline", "en", "clinical/models")

pipeline.annotate("A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25-year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.")
```
</div>

## Results

```bash
+-----------------------------+------------+
|chunks                       |entities    |
+-----------------------------+------------+
|2093-01-13                   |DATE        |
|David Hale                   |DOCTOR      |
|Hendrickson, Ora             |DOCTOR      |
|7194334                      |PHONE       |
|01/13/93                     |DATE        |
|Oliveira                     |DOCTOR      |
|1-11-2000                    |DATE        |
|Cocke County Baptist Hospital|HOSPITAL    |
|0295 Keats Street            |STREET      |
|(302) 786-5227               |PHONE       |
|Brothers Coal-Mine           |ORGANIZATION|
+-----------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_enriched_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.0 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
