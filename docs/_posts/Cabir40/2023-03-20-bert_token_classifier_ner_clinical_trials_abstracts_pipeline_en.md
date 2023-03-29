---
layout: model
title: Pipeline to Extract entities in clinical trial abstracts (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_clinical_trials_abstracts_pipeline
date: 2023-03-20
tags: [berttokenclassifier, bert, biobert, en, ner, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_clinical_trials_abstracts](https://nlp.johnsnowlabs.com/2022/06/29/bert_token_classifier_ner_clinical_trials_abstracts_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_clinical_trials_abstracts_pipeline_en_4.3.0_3.2_1679304059319.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_clinical_trials_abstracts_pipeline_en_4.3.0_3.2_1679304059319.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_clinical_trials_abstracts_pipeline", "en", "clinical/models")

text = '''This open-label, parallel-group, two-arm, pilot study compared the beta-cell protective effect of adding insulin glargine (GLA) vs. NPH insulin to ongoing metformin. Overall, 28 insulin-naive type 2 diabetes subjects (mean +/- SD age, 61.5 +/- 6.7 years; BMI, 30.7 +/- 4.3 kg/m(2)) treated with metformin and sulfonylurea were randomized to add once-daily GLA or NPH at bedtime.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_clinical_trials_abstracts_pipeline", "en", "clinical/models")

val text = "This open-label, parallel-group, two-arm, pilot study compared the beta-cell protective effect of adding insulin glargine (GLA) vs. NPH insulin to ongoing metformin. Overall, 28 insulin-naive type 2 diabetes subjects (mean +/- SD age, 61.5 +/- 6.7 years; BMI, 30.7 +/- 4.3 kg/m(2)) treated with metformin and sulfonylurea were randomized to add once-daily GLA or NPH at bedtime."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk        |   begin |   end | ner_label          |   confidence |
|---:|:-----------------|--------:|------:|:-------------------|-------------:|
|  0 | open-label       |       5 |    14 | CTDesign           |     0.742075 |
|  1 | parallel-group   |      17 |    30 | CTDesign           |     0.725741 |
|  2 | two-arm          |      33 |    39 | CTDesign           |     0.427547 |
|  3 | insulin glargine |     105 |   120 | Drug               |     0.985063 |
|  4 | GLA              |     123 |   125 | Drug               |     0.96917  |
|  5 | NPH insulin      |     132 |   142 | Drug               |     0.762519 |
|  6 | metformin        |     155 |   163 | Drug               |     0.996344 |
|  7 | 28               |     175 |   176 | NumberPatients     |     0.968501 |
|  8 | type 2 diabetes  |     192 |   206 | DisorderOrSyndrome |     0.979685 |
|  9 | 61.5             |     235 |   238 | Age                |     0.610416 |
| 10 | kg/m(2           |     273 |   278 | BioAndMedicalUnit  |     0.974807 |
| 11 | metformin        |     295 |   303 | Drug               |     0.99696  |
| 12 | sulfonylurea     |     309 |   320 | Drug               |     0.996722 |
| 13 | randomized       |     327 |   336 | CTDesign           |     0.990632 |
| 14 | once-daily       |     345 |   354 | DrugTime           |     0.472084 |
| 15 | GLA              |     356 |   358 | Drug               |     0.972978 |
| 16 | NPH              |     363 |   365 | Drug               |     0.989424 |
| 17 | bedtime          |     370 |   376 | DrugTime           |     0.936016 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_clinical_trials_abstracts_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel