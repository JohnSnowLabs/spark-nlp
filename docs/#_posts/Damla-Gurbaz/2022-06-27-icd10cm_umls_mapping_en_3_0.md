---
layout: model
title: Pipeline to Mapping ICD10-CM Codes with Their Corresponding UMLS Codes
author: John Snow Labs
name: icd10cm_umls_mapping
date: 2022-06-27
tags: [icd10cm, umls, pipeline, chunk_mapper, clinical, licensed, en]
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

This pretrained pipeline is built on the top of `icd10cm_umls_mapper` model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/icd10cm_umls_mapping_en_3.5.3_3.0_1656366054366.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("icd10cm_umls_mapping", "en", "clinical/models")

result = pipeline.fullAnnotate(["M8950", "R822", "R0901"])
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("icd10cm_umls_mapping", "en", "clinical/models")

val result = pipeline.fullAnnotate(Array("M8950", "R822", "R0901"))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.icd10cm.umls").predict("""Put your text here.""")
```

</div>

## Results

```bash
|    | icd10cm_code   | umls_code   |
|---:|:---------------|:------------|
|  0 | M8950          | C4721411    |
|  1 | R822           | C0159076    |
|  2 | R0901          | C0004044    |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|icd10cm_umls_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|952.4 KB|

## Included Models

- DocumentAssembler
- TokenizerModel
- ChunkMapperModel