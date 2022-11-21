---
layout: model
title: ICD10 to ICD9 Code Mapping
author: John Snow Labs
name: icd10_icd9_mapping
date: 2021-12-22
tags: [icd10, icd9, en, clinical, licensed, code_mapping]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps ICD10 codes to ICD9 codes without using any text data. You’ll just feed a comma or white space-delimited ICD10 codes and it will return the corresponding ICD9 codes as a list.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.1.Healthcare_Code_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10_icd9_mapping_en_3.3.4_2.4_1640175449509.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("icd10_icd9_mapping", "en", "clinical/models")
pipeline.annotate('E669 R630 J988')
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("icd10_icd9_mapping", "en", "clinical/models")
val result = pipeline.annotate('E669 R630 J988')
```
</div>

## Results

```bash
{'document': ['E669 R630 J988'],
'icd10': ['E669', 'R630', 'J988'],
'icd9': ['27800', '7830', '5198']}


Note:

| ICD10 | Details |
| ---------- | ----------------------------:|
| E669 | Obesity |
| R630 | Anorexia |
| J988 | Other specified respiratory disorders |

| ICD9 | Details |
| ---------- | ---------------------------:|
| 27800 | Obesity |
| 7830 | Anorexia |
| 5198 | Other diseases of respiratory system |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10_icd9_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|545.2 KB|

## Included Models

- DocumentAssembler
- TokenizerModel
- LemmatizerModel