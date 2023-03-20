---
layout: model
title: Pipeline to Detect PHI in text (enriched-biobert)
author: John Snow Labs
name: ner_deid_enriched_biobert_pipeline
date: 2023-03-20
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_biobert_pipeline_en_4.3.0_3.2_1679316429600.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_biobert_pipeline_en_4.3.0_3.2_1679316429600.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_deid_enriched_biobert_pipeline", "en", "clinical/models")

text = '''A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_deid_enriched_biobert_pipeline", "en", "clinical/models")

val text = "A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                     |   begin |   end | ner_label    |   confidence |
|---:|:------------------------------|--------:|------:|:-------------|-------------:|
|  0 | 2093-01-13                    |      17 |    26 | DATE         |     0.9267   |
|  1 | David Hale                    |      29 |    38 | DOCTOR       |     0.7949   |
|  2 | Hendrickson, Ora              |      53 |    68 | PATIENT      |     0.637733 |
|  3 | 7194334                       |      76 |    82 | PHONE        |     0.4939   |
|  4 | Cocke County Baptist Hospital |     114 |   142 | HOSPITAL     |     0.6199   |
|  5 | 0295 Keats Street             |     145 |   161 | STREET       |     0.592433 |
|  6 | 302) 786-5227                 |     174 |   186 | PHONE        |     0.846833 |
|  7 | Brothers Coal-Mine            |     253 |   270 | ORGANIZATION |     0.45085  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_enriched_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel