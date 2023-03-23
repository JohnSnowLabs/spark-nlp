---
layout: model
title: Pipeline to Detect posology entities (biobert)
author: John Snow Labs
name: ner_posology_biobert_pipeline
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

This pretrained pipeline is built on the top of [ner_posology_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_biobert_pipeline_en_4.3.0_3.2_1679316307940.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_posology_biobert_pipeline_en_4.3.0_3.2_1679316307940.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_posology_biobert_pipeline", "en", "clinical/models")

text = '''The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_posology_biobert_pipeline", "en", "clinical/models")

val text = "The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk           |   begin |   end | ner_label   |   confidence |
|---:|:--------------------|--------:|------:|:------------|-------------:|
|  0 | 1                   |      27 |    27 | DOSAGE      |     0.9993   |
|  1 | capsule             |      29 |    35 | FORM        |     0.9998   |
|  2 | Advil               |      40 |    44 | DRUG        |     0.9999   |
|  3 | 10 mg               |      46 |    50 | STRENGTH    |     0.98145  |
|  4 | for 5 days          |      52 |    61 | DURATION    |     0.998833 |
|  5 | magnesium hydroxide |      67 |    85 | DRUG        |     0.82655  |
|  6 | 100mg/1ml           |      87 |    95 | STRENGTH    |     0.9391   |
|  7 | PO                  |     108 |   109 | ROUTE       |     1        |
|  8 | 40 units            |     179 |   186 | DOSAGE      |     0.87745  |
|  9 | insulin glargine    |     191 |   206 | DRUG        |     0.9817   |
| 10 | at night            |     208 |   215 | FREQUENCY   |     0.8641   |
| 11 | 12 units            |     218 |   225 | DOSAGE      |     0.9533   |
| 12 | insulin lispro      |     230 |   243 | DRUG        |     0.9476   |
| 13 | with meals          |     245 |   254 | FREQUENCY   |     0.82125  |
| 14 | metformin           |     261 |   269 | DRUG        |     0.9999   |
| 15 | 1000 mg             |     271 |   277 | STRENGTH    |     0.91255  |
| 16 | two times a day     |     279 |   293 | FREQUENCY   |     0.9969   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_posology_biobert_pipeline|
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