---
layout: model
title: Clinical Ner Assertion
author: John Snow Labs
name: clinical_ner_assertion
class: PipelineModel
language: en
repository: clinical/models
date: 2020-01-31
tags: [ner, clinical, licensed]
supported: true
edition: Spark NLP 2.4.0
spark_version: 2.4
annotator: PipelineModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
A pretrained pipeline with ``ner_clinical`` and ``assertion_dl``. It will extract clinical entities and assign assertion status for them.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_ner_assertion_en_2.4.0_2.4_1580481098096.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/clinical_ner_assertion_en_2.4.0_2.4_1580481098096.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
pipeline = PretrainedPipeline("clinical_ner_assertion","en","clinical/models")

result = pipe_model.fullAnnotate("""She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed. She denied pain and any headache.She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge.""")[0]

result.keys()
```

```scala
val pipeline = new PretrainedPipeline("clinical_ner_assertion","en","clinical/models")

val result = pipeline.fullAnnotate("She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed. She denied pain and any headache.She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge.")(0)
```
</div>

{:.h2_title}
## Results
```bash

                          chunks  entities  assertion

0  gestational diabetes mellitus   PROBLEM  present
1                           pain   PROBLEM  absent
2                       headache   PROBLEM  absent
3               insulin glargine TREATMENT  present
4                 insulin lispro TREATMENT  present
5                      metformin TREATMENT  present
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|------------------------|
| Name:          | clinical_ner_assertion |
| Type:   | PipelineModel          |
| Compatibility: | Spark NLP 2.4.0+                  |
| License:       | Licensed               |
| Edition:       | Official             |
| Language:      | en                     |


{:.h2_title}
## Included Models 
 - ner_clinical
 - assertion_dl
