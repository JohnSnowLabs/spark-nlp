---
layout: model
title: Explain Document Pipeline - CRA
author: John Snow Labs
name: explain_clinical_doc_cra
class: PipelineModel
language: en
repository: clinical/models
date: 2020-08-19
tags: [clinical,pipeline,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
A pretrained pipeline with `ner_clinical`, `assertion_dl`, `re_clinical`. It will extract clinical entities, assign assertion status and find relationships between clinical entities.


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_cra_en_2.5.5_2.4_1597846145640.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
cra_pipeline = PretrainedPipeline("explain_clinical_doc_cra","en","clinical/models")

annotations =  cra_pipeline.fullAnnotate("""She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed. She denied pain and any headache.She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge. 
""")[0]

annotations.keys()

```

```scala

val cra_pipeline = new PretrainedPipeline("explain_clinical_doc_cra","en","clinical/models")

val result = cra_pipeline.fullAnnotate("""She is admitted to The John Hopkins Hospital 2 days ago with a history of gestational diabetes mellitus diagnosed. She denied pain and any headache.She was seen by the endocrinology service and she was discharged on 03/02/2018 on 40 units of insulin glargine, 12 units of insulin lispro, and metformin 1000 mg two times a day. She had close follow-up with endocrinology post discharge. 
""")(0)

```
</div>

{:.h2_title}
## Results
This pretrained pipeline gives the result of `ner_clinical`, `re_clinical` and `assertion_dl` models. Here is the result of `clinical_ner_chunks` and `assertion`:

```bash
|   | chunk                         | ner_clinical | assertion | relation |
|---|-------------------------------|--------------|-----------|----------|
| 0 | gestational diabetes mellitus | PROBLEM      | present   | BEFORE   |
| 1 | pain                          | PROBLEM      | absent    | OVERLAP  |
| 2 | insulin glargine              | TREATMENT    | present   | OVERLAP  |
| 3 | metformin                     | TREATMENT    | present   | OVERLAP  |

```

{:.model-param}
## Model Information

{:.table-model}
|---------------|--------------------------|
| Name:          | explain_clinical_doc_cra |
| Type:   | PipelineModel            |
| Compatibility: | Spark NLP 2.5.5+                    |
| License:       | Licensed                 |
| Edition:       | Official               |
| Language:      | en                       |


{:.h2_title}
## Included Models
- ner_clinical
- assertion_dl
- re_clinical
