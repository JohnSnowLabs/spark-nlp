---
layout: model
title: Pipeline to find clinical events and find temporal relations (ERA)
author: John Snow Labs
name: explain_clinical_doc_era
date: 2021-04-01
tags: [pipeline, en, licensed, clinical]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pipeline with `ner_clinical_events`, `assertion_dl` and `re_temporal_events_clinical`. It will extract clinical entities, assign assertion status and find temporal relationships between clinical entities.

{:.btn-box}<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_era_en_3.0.0_3.0_1617297404938.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_era_en_3.0.0_3.0_1617297404938.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
era_pipeline = PretrainedPipeline('explain_clinical_doc_era', 'en', 'clinical/models')

annotations =  era_pipeline.fullAnnotate("""She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed. She denied pain and any headache. She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge. """)[0]

```
```scala
val era_pipeline = new PretrainedPipeline("explain_clinical_doc_era", "en", "clinical/models")

val result = era_pipeline.fullAnnotate("""She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed. She denied pain and any headache. She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge. """)(0)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.explain_doc.era").predict("""She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed. She denied pain and any headache. She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge. """)
```

</div>

## Results

```bash
|    | relation   | entity1       |   entity1_begin |   entity1_end | chunk1                    | entity2       |   entity2_begin |   entity2_end | chunk2                        |   confidence |
|---:|:-----------|:--------------|----------------:|--------------:|:--------------------------|:--------------|----------------:|--------------:|:------------------------------|-------------:|
|  0 | AFTER      | OCCURRENCE    |               7 |            14 | admitted                  | CLINICAL_DEPT |              19 |            43 | The John Hopkins Hospital     |     0.963836 |
|  1 | BEFORE     | OCCURRENCE    |               7 |            14 | admitted                  | DATE          |              45 |            54 | 2 days ago                    |     0.587098 |
|  2 | BEFORE     | OCCURRENCE    |               7 |            14 | admitted                  | PROBLEM       |              74 |           102 | gestational diabetes mellitus |     0.999991 |
|  3 | OVERLAP    | CLINICAL_DEPT |              19 |            43 | The John Hopkins Hospital | DATE          |              45 |            54 | 2 days ago                    |     0.996056 |
|  4 | BEFORE     | CLINICAL_DEPT |              19 |            43 | The John Hopkins Hospital | PROBLEM       |              74 |           102 | gestational diabetes mellitus |     0.995216 |
|  5 | OVERLAP    | DATE          |              45 |            54 | 2 days ago                | PROBLEM       |              74 |           102 | gestational diabetes mellitus |     0.996954 |
|  6 | BEFORE     | EVIDENTIAL    |             119 |           124 | denied                    | PROBLEM       |             126 |           129 | pain                          |     1        |
|  7 | BEFORE     | EVIDENTIAL    |             119 |           124 | denied                    | PROBLEM       |             135 |           146 | any headache                  |     1        |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_era|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|