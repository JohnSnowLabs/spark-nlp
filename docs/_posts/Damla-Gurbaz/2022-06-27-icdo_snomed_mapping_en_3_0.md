---
layout: model
title: Pipeline to Mapping ICDO Codes with Their Corresponding SNOMED Codes
author: John Snow Labs
name: icdo_snomed_mapping
date: 2022-06-27
tags: [icdo, snomed, pipeline, chunk_mapper, clinical, licensed, en]
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

This pretrained pipeline is built on the top of `icdo_snomed_mapper` model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icdo_snomed_mapping_en_3.5.3_3.0_1656364275328.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/icdo_snomed_mapping_en_3.5.3_3.0_1656364275328.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("icdo_snomed_mapping", "en", "clinical/models")

result= pipeline.fullAnnotate("8120/1 8170/3 8380/3")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline= new PretrainedPipeline("icdo_snomed_mapping", "en", "clinical/models")

val result= pipeline.fullAnnotate("8120/1 8170/3 8380/3")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.icdo_to_snomed.pipe").predict("""8120/1 8170/3 8380/3""")
```

</div>

## Results

```bash
|    | icdo_code                | snomed_code                    |
|---:|:-------------------------|:-------------------------------|
|  0 | 8120/1 | 8170/3 | 8380/3 | 45083001 | 25370001 | 30289006 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icdo_snomed_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|133.2 KB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ChunkMapperModel
