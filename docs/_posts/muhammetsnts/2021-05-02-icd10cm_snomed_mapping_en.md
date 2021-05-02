---
layout: model
title: ICD10 to Snomed Code Mapping
author: John Snow Labs
name: icd10cm_snomed_mapping
date: 2021-05-02
tags: [icd10cm, snomed, en, licensed]
task: Pipeline Healthcare
language: en
edition: Spark NLP 3.0.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps ICD10CM codes to SNOMED codes without using any text data. You’ll just feed a comma or white space-delimited ICD10CM codes and it will return the corresponding SNOMED codes as a list. For the time being, it supports 132K Snomed codes and will be augmented & enriched in the next releases.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10cm_snomed_mapping_en_3.0.2_3.0_1619954408986.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline( "icd10cm_snomed_mapping","en","clinical/models")
pipeline.annotate('M89.50 I288 H16269')
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline("icd10cm_snomed_mapping","en","clinical/models")
val result = pipeline.annotate('M89.50 I288 H16269')
```
</div>

## Results

```bash
{'icd10cm': ['M89.50', 'I288', 'H16269'],
 'snomed': ['733187009', '449433008', '51264003']}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10cm_snomed_mapping|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- TokenizerModel
- LemmatizerModel
- Finisher