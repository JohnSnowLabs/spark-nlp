---
layout: model
title: MeSH to UMLS Code Mapping
author: John Snow Labs
name: mesh_umls_mapping
date: 2021-07-01
tags: [mesh, umls, en, licensed, pipeline]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.1.0
spark_version: 2.4
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline maps MeSH codes to UMLS codes without using any text data. You’ll just feed white space-delimited MeSH codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_CODE_MAPPING/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/mesh_umls_mapping_en_3.1.0_2.4_1625126292100.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/mesh_umls_mapping_en_3.1.0_2.4_1625126292100.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline("mesh_umls_mapping","en","clinical/models")
pipeline.annotate("C028491 D019326 C579867")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline("mesh_umls_mapping","en","clinical/models")
val result = pipeline.annotate("C028491 D019326 C579867")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.mesh.umls").predict("""C028491 D019326 C579867""")
```

</div>

## Results

```bash
{'mesh': ['C028491', 'D019326', 'C579867'],
'umls': ['C0970275', 'C0886627', 'C3696376']}

Note:

| MeSH       | Details                      | 
| ---------- | ----------------------------:|
| C028491    |  1,3-butylene glycol         |
| D019326    | 17-alpha-Hydroxyprogesterone |
| C579867    | 3-Methylglutaconic Aciduria  |

| UMLS       | Details                     |
| ---------- | ---------------------------:|
| C0970275   | 1,3-butylene glycol         |
| C0886627   | 17-hydroxyprogesterone      |
| C3696376   | 3-methylglutaconic aciduria |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mesh_umls_mapping|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.1.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- TokenizerModel
- LemmatizerModel
- Finisher