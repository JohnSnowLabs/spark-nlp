---
layout: model
title: Pipeline to Mapping MESH Codes with Their Corresponding UMLS Codes
author: John Snow Labs
name: mesh_umls_mapping
date: 2022-06-27
tags: [mesh, umls, chunk_mapper, pipeline, clinical, licensed, en]
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

This pretrained pipeline is built on the top of `mesh_umls_mapper` model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/mesh_umls_mapping_en_3.5.3_3.0_1656366727552.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/mesh_umls_mapping_en_3.5.3_3.0_1656366727552.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline= PretrainedPipeline("mesh_umls_mapping", "en", "clinical/models")

result = pipeline.fullAnnotate("C028491 D019326 C579867")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline= new PretrainedPipeline("mesh_umls_mapping", "en", "clinical/models")

val result = pipeline.fullAnnotate("C028491 D019326 C579867")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.mesh.umls").predict("""C028491 D019326 C579867""")
```

</div>

## Results

```bash
|    | mesh_code                   | umls_code                      |
|---:|:----------------------------|:-------------------------------|
|  0 | C028491 | D019326 | C579867 | C0043904 | C0045010 | C3696376 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mesh_umls_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|3.8 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ChunkMapperModel