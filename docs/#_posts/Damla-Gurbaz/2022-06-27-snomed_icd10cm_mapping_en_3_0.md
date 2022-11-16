---
layout: model
title: Pipeline to Mapping SNOMED Codes with Their Corresponding ICD10-CM Codes
author: John Snow Labs
name: snomed_icd10cm_mapping
date: 2022-06-27
tags: [pipeline, snomed, icd10cm, chunk_mapper, clinical, licensed, en]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of `snomed_icd10cm_mapper` model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/snomed_icd10cm_mapping_en_3.5.3_3.0_1656363315439.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("snomed_icd10cm_mapping", "en", "clinical/models")

result= pipeline.fullAnnotate("128041000119107 292278006 293072005")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline= new PretrainedPipeline("snomed_icd10cm_mapping", "en", "clinical/models")

val result= pipeline.fullAnnotate("128041000119107 292278006 293072005")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.snomed_to_icd10cm.pipe").predict("""128041000119107 292278006 293072005""")
```

</div>

## Results

```bash
|    | snomed_code                             | icd10cm_code               |
|---:|:----------------------------------------|:---------------------------|
|  0 | 128041000119107 | 292278006 | 293072005 | K22.70 | T43.595 | T37.1X5 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|snomed_icd10cm_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.5 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ChunkMapperModel