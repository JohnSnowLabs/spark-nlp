---
layout: model
title: Pipeline to Detect PHI for Deidentification (Subentity- Augmented)
author: John Snow Labs
name: ner_deid_subentity_augmented_pipeline
date: 2023-03-13
tags: [deid, ner, en, i2b2, licensed]
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

This pretrained pipeline is built on the top of [ner_deid_subentity_augmented](https://nlp.johnsnowlabs.com/2021/09/03/ner_deid_subentity_augmented_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_augmented_pipeline_en_4.3.0_3.2_1678734896498.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_augmented_pipeline_en_4.3.0_3.2_1678734896498.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_deid_subentity_augmented_pipeline", "en", "clinical/models")

text = '''Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson Ora , MR # 7194334 Date : 01/13/93 . PCP : Oliveira , 25 years old , Record date : 2079-11-09 . Cocke County Baptist Hospital , 0295 Keats Street , Phone 302-786-5227.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_deid_subentity_augmented_pipeline", "en", "clinical/models")

val text = "Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson Ora , MR # 7194334 Date : 01/13/93 . PCP : Oliveira , 25 years old , Record date : 2079-11-09 . Cocke County Baptist Hospital , 0295 Keats Street , Phone 302-786-5227."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                    |   begin |   end | ner_label     |   confidence |
|---:|:------------------------------|--------:|------:|:--------------|-------------:|
|  0 | 2093-01-13                    |      14 |    23 | DATE          |      1       |
|  1 | David Hale                    |      27 |    36 | DOCTOR        |      0.97385 |
|  2 | Hendrickson Ora               |      55 |    69 | PATIENT       |      0.9932  |
|  3 | 7194334                       |      78 |    84 | MEDICALRECORD |      0.9993  |
|  4 | 01/13/93                      |      93 |   100 | DATE          |      1       |
|  5 | Oliveira                      |     110 |   117 | DOCTOR        |      0.9993  |
|  6 | 25                            |     121 |   122 | AGE           |      0.9905  |
|  7 | 2079-11-09                    |     150 |   159 | DATE          |      0.9998  |
|  8 | Cocke County Baptist Hospital |     163 |   191 | HOSPITAL      |      0.97485 |
|  9 | 0295 Keats Street             |     195 |   211 | STREET        |      0.8209  |
| 10 | 302-786-5227                  |     221 |   232 | PHONE         |      0.9541  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity_augmented_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
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
- NerConverterInternalModel