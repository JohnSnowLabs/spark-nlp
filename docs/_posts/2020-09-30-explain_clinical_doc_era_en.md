---
layout: model
title: Detect Clinical Entities, Assign Assertion and Find Relations
author: John Snow Labs
name: explain_clinical_doc_era
date: 2020-09-30
task: [Named Entity Recognition, Assertion Status, Relation Extraction, Pipeline Healthcare]
language: en
edition: Healthcare NLP 2.6.0
spark_version: 2.4
tags: [pipeline, en, licensed, clinical]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
A pretrained pipeline with ``ner_clinical_events``, ``assertion_dl`` and ``re_temporal_events_clinical`` trained with ``embeddings_healthcare_100d``. It will extract clinical entities, assign assertion status and find temporal relationships between clinical entities.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_era_en_2.5.5_2.4_1597845753750.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_era_en_2.5.5_2.4_1597845753750.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python

era_pipeline = PretrainedPipeline('explain_clinical_doc_era', 'en', 'clinical/models')

annotations =  era_pipeline.fullAnnotate("""She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed. She denied pain and any headache. She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge. """)[0]

annotations.keys()

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

{:.h2_title}
## Results
The output is a dictionary with the following keys: ``'sentences'``, ``'clinical_ner_tags'``, ``'clinical_ner_chunks_re'``, ``'document'``, ``'clinical_ner_chunks'``, ``'assertion'``, ``'clinical_relations'``, ``'tokens'``, ``'embeddings'``, ``'pos_tags'``, ``'dependencies'``. Here is the result of `clinical_ner_chunks` :
```bash
| #  | chunks                        | begin | end | entities      |
|----|-------------------------------|-------|-----|---------------|
| 0  | admitted                      | 7     | 14  | OCCURRENCE    |
| 1  | The John Hopkins Hospital     | 19    | 43  | CLINICAL_DEPT |
| 2  | 2 days ago                    | 45    | 54  | DATE          |
| 3  | gestational diabetes mellitus | 74    | 102 | PROBLEM       |
| 4  | diagnosed                     | 104   | 112 | OCCURRENCE    |
| 5  | denied                        | 119   | 124 | EVIDENTIAL    |
| 6  | pain                          | 126   | 129 | PROBLEM       |
| 7  | any headache                  | 135   | 146 | PROBLEM       |
| 8  | seen                          | 157   | 160 | OCCURRENCE    |
| 9  | the endocrinology service     | 165   | 189 | CLINICAL_DEPT |
| 10 | discharged                    | 203   | 212 | OCCURRENCE    |
| 11 | 03/02/2018                    | 217   | 226 | DATE          |
| 12 | insulin glargine              | 243   | 258 | TREATMENT     |
| 13 | insulin lispro                | 274   | 287 | TREATMENT     |
| 14 | metformin                     | 294   | 302 | TREATMENT     |
| 15 | two times a day               | 312   | 326 | FREQUENCY     |
| 16 | close follow-up               | 337   | 351 | OCCURRENCE    |
| 17 | endocrinology                 | 358   | 370 | CLINICAL_DEPT |
| 18 | discharge                     | 377   | 385 | OCCURRENCE    |
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_era|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 2.6.0 +|
|License:|Licensed|
|Edition:|Official|
|Language:|[en]|

{:.h2_title}
## Included Models 
- ``ner_clinical_events``
- ``assertion_dl``
- ``re_temporal_events_clinical``

