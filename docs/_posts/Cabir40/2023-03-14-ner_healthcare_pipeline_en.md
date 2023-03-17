---
layout: model
title: Pipeline to Detect Problems, Tests and Treatments
author: John Snow Labs
name: ner_healthcare_pipeline
date: 2023-03-14
tags: [ner, licensed, clinical, en]
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

This pretrained pipeline is built on the top of [ner_healthcare](https://nlp.johnsnowlabs.com/2021/04/21/ner_healthcare_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_pipeline_en_4.3.0_3.2_1678824932575.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_pipeline_en_4.3.0_3.2_1678824932575.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_healthcare_pipeline", "en", "clinical/models")

text = '''A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG .'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_healthcare_pipeline", "en", "clinical/models")

val text = "A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG ."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                    |   begin |   end | ner_label   |   confidence |
|---:|:------------------------------|--------:|------:|:------------|-------------:|
|  0 | gestational diabetes mellitus |      39 |    67 | PROBLEM     |     0.938233 |
|  1 | type two diabetes mellitus    |     128 |   153 | PROBLEM     |     0.762925 |
|  2 | HTG-induced pancreatitis      |     186 |   209 | PROBLEM     |     0.9742   |
|  3 | an acute hepatitis            |     263 |   280 | PROBLEM     |     0.915067 |
|  4 | obesity                       |     288 |   294 | PROBLEM     |     0.9926   |
|  5 | a body mass index             |     301 |   317 | TEST        |     0.721175 |
|  6 | BMI                           |     321 |   323 | TEST        |     0.4466   |
|  7 | polyuria                      |     380 |   387 | PROBLEM     |     0.9987   |
|  8 | polydipsia                    |     391 |   400 | PROBLEM     |     0.9993   |
|  9 | poor appetite                 |     404 |   416 | PROBLEM     |     0.96315  |
| 10 | vomiting                      |     424 |   431 | PROBLEM     |     0.9588   |
| 11 | amoxicillin                   |     511 |   521 | TREATMENT   |     0.6453   |
| 12 | a respiratory tract infection |     527 |   555 | PROBLEM     |     0.867    |
| 13 | metformin                     |     570 |   578 | TREATMENT   |     0.9989   |
| 14 | glipizide                     |     582 |   590 | TREATMENT   |     0.9997   |
| 15 | dapagliflozin                 |     598 |   610 | TREATMENT   |     0.9996   |
| 16 | T2DM                          |     616 |   619 | TREATMENT   |     0.9662   |
| 17 | atorvastatin                  |     625 |   636 | TREATMENT   |     0.9993   |
| 18 | gemfibrozil                   |     642 |   652 | TREATMENT   |     0.9997   |
| 19 | HTG                           |     658 |   660 | PROBLEM     |     0.9927   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|513.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel