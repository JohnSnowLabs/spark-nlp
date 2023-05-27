---
layout: model
title: Pipeline to Mapping ICD10-CM Codes with Their Corresponding SNOMED Codes
author: John Snow Labs
name: icd10cm_snomed_mapping
date: 2023-05-27
tags: [en, licensed, icd10cm, snomed, pipeline, chunk_mapping, open_source]
task: Chunk Mapping
language: en
edition: Healthcare NLP 4.3.1
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of `icd10cm_snomed_mapper` model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/icd10cm_snomed_mapping_en_4.3.1_3.4_1685183538520.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/icd10cm_snomed_mapping_en_4.3.1_3.4_1685183538520.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("icd10cm_snomed_mapping", "en", "clinical/models")

result = pipeline.fullAnnotate(R079 N4289 M62830)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("icd10cm_snomed_mapping", "en", "clinical/models")

val result = pipeline.fullAnnotate(R079 N4289 M62830)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.icd10cm_to_snomed.pipe").predict("""Put your text here.""")
```

</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("icd10cm_snomed_mapping", "en", "clinical/models")

result = pipeline.fullAnnotate(R079 N4289 M62830)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("icd10cm_snomed_mapping", "en", "clinical/models")

val result = pipeline.fullAnnotate(R079 N4289 M62830)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.icd10cm_to_snomed.pipe").predict("""Put your text here.""")
```
</div>

## Results

```bash
Results



|    | icd10cm_code          | snomed_code                              |
|---:|:----------------------|:-----------------------------------------|
|  0 | R079 | N4289 | M62830 | 161972006 | 22035000 | 16410651000119105 |



{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10cm_snomed_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.1 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ChunkMapperModel