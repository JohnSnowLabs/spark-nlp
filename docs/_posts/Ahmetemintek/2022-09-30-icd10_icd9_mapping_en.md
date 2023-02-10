---
layout: model
title: Pipeline to Mapping ICD10-CM Codes with Their Corresponding ICD-9-CM Codes
author: John Snow Labs
name: icd10_icd9_mapping
date: 2022-09-30
tags: [icd10cm, icd9cm, licensed, en, clinical, pipeline, chunk_mapping]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.1.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of icd10_icd9_mapper model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10_icd9_mapping_en_4.1.0_3.0_1664539288823.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/icd10_icd9_mapping_en_4.1.0_3.0_1664539288823.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("icd10_icd9_mapping", "en", "clinical/models")

result= pipeline.fullAnnotate('Z833 A0100 A000')
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline= new PretrainedPipeline("icd10_icd9_mapping", "en", "clinical/models")

val result= pipeline.fullAnnotate('Z833 A0100 A000')
```
</div>

## Results

```bash
|    | icd10_code          | icd9_code          |
|---:|:--------------------|:-------------------|
|  0 | Z833 | A0100 | A000 | V180 | 0020 | 0010 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10_icd9_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|589.5 KB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ChunkMapperModel