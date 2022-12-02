---
layout: model
title: Pipeline to Mapping SNOMED Codes with Their Corresponding UMLS Codes
author: John Snow Labs
name: snomed_umls_mapping
date: 2022-06-27
tags: [snomed, umls, pipeline, chunk_mapper, clinical, licensed, en]
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

This pretrained pipeline is built on the top of `snomed_umls_mapper` model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/snomed_umls_mapping_en_3.5.3_3.0_1656368000448.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("snomed_umls_mapping", "en", "clinical/models")

result= pipeline.fullAnnotate("733187009 449433008 51264003")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("snomed_umls_mapping", "en", "clinical/models")

val result= pipeline.fullAnnotate("733187009 449433008 51264003")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.snomed.umls").predict("""733187009 449433008 51264003""")
```

</div>

## Results

```bash
|    | snomed_code                      | umls_code                      |
|---:|:---------------------------------|:-------------------------------|
|  0 | 733187009 | 449433008 | 51264003 | C4546029 | C3164619 | C0271267 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|snomed_umls_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|5.1 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ChunkMapperModel