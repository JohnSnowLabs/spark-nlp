---
layout: model
title: Pipeline to Mapping ICD10-CM Codes with Their Corresponding SNOMED Codes
author: John Snow Labs
name: icd10cm_snomed_mapping
date: 2022-06-27
tags: [icd10cm, snomed, pipeline, clinical, en, licensed, chunk_mapper]
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

This pretrained pipeline is built on the top of `icd10cm_snomed_mapper` model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10cm_snomed_mapping_en_3.5.3_3.0_1656361159581.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/icd10cm_snomed_mapping_en_3.5.3_3.0_1656361159581.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("icd10cm_snomed_mapping", "en", "clinical/models")

result= pipeline.fullAnnotate('R079 N4289 M62830')
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline= new PretrainedPipeline("icd10cm_snomed_mapping", "en", "clinical/models")

val result= pipeline.fullAnnotate("R079 N4289 M62830")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.icd10cm_to_snomed.pipe").predict("""R079 N4289 M62830""")
```

</div>

## Results

```bash
|    | icd10cm_code          | snomed_code                              |
|---:|:----------------------|:-----------------------------------------|
|  0 | R079 | N4289 | M62830 | 161972006 | 22035000 | 16410651000119105 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10cm_snomed_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.1 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ChunkMapperModel