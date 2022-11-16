---
layout: model
title: RxNorm to MeSH Code Mapping
author: John Snow Labs
name: rxnorm_mesh_mapping
date: 2021-05-04
tags: [rxnorm, mesh, en, licensed]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.0.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps RxNorm codes to MeSH codes without using any text data. You’ll just feed white space-delimited RxNorm codes and it will return the corresponding MeSH codes as a list. If there is no mapping, the original code is returned with no mapping.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rxnorm_mesh_mapping_en_3.0.2_2.4_1620134962818.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("rxnorm_mesh_mapping","en","clinical/models")
pipeline.annotate("1191 6809 47613")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline("rxnorm_mesh_mapping","en","clinical/models")
val result = pipeline.annotate("1191 6809 47613")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.rxnorm.mesh").predict("""1191 6809 47613""")
```

</div>

## Results

```bash
{'rxnorm': ['1191', '6809', '47613'],
'mesh': ['D001241', 'D008687', 'D019355']}


Note:

| RxNorm     | Details             |
| ---------- | -------------------:|
| 1191       |  aspirin            |
| 6809       | metformin           |
| 47613      | calcium citrate     |

| MeSH       | Details             |
| ---------- | -------------------:|
| D001241    | Aspirin             |
| D008687    | Metformin           |
| D019355    | Calcium Citrate     |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rxnorm_mesh_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.0.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- TokenizerModel
- LemmatizerModel
- Finisher