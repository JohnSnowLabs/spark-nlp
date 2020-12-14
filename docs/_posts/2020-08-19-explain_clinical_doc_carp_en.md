---
layout: model
title: Explain Document Pipeline - CARP
author: John Snow Labs
name: explain_clinical_doc_carp
date: 2020-08-19
tags: [pipeline, en, clinical, licensed]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
A pretrained pipeline with ``ner_clinical``, ``assertion_dl``, ``re_clinical`` and ``ner_posology``. It will extract clinical and medication entities, assign assertion status and find relationships between clinical entities.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/explain_clinical_doc_carp_en_2.5.5_2.4_1597841630062.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
carp_pipeline = PretrainedPipeline("explain_clinical_doc_carp","en","clinical/models")

annotations =  carp_pipeline.fullAnnotate("""A 28-year-old female with a history of gestational diabetes mellitus, used to take metformin 1000 mg two times a day, presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting. She was seen by the endocrinology service and discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals.""")[0]

annotations.keys()

```

```scala

val carp_pipeline = new PretrainedPipeline("explain_clinical_doc_carp","en","clinical/models")

val result = carp_pipeline.fullAnnotate("""A 28-year-old female with a history of gestational diabetes mellitus, used to take metformin 1000 mg two times a day, presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting. She was seen by the endocrinology service and discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals.""")(0)

```
</div>

{:.h2_title}
## Results
This pretrained pipeline gives the result of `ner_clinical`, `re_clinical`, `ner_posology` and `assertion_dl` models. 
```bash
|   | chunks                        | ner_clinical | assertion | posology_chunk   | ner_posology | relations |
|---|-------------------------------|--------------|-----------|------------------|--------------|-----------|
| 0 | gestational diabetes mellitus | PROBLEM      | present   | metformin        | Drug         | TrAP      |
| 1 | metformin                     | TREATMENT    | present   | 1000 mg          | Strength     | TrCP      |
| 2 | polyuria                      | PROBLEM      | present   | two times a day  | Frequency    | TrCP      |
| 3 | polydipsia                    | PROBLEM      | present   | 40 units         | Dosage       | TrWP      |
| 4 | poor appetite                 | PROBLEM      | present   | insulin glargine | Drug         | TrCP      |
| 5 | vomiting                      | PROBLEM      | present   | at night         | Frequency    | TrAP      |
| 6 | insulin glargine              | TREATMENT    | present   | 12 units         | Dosage       | TrAP      |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_clinical_doc_carp|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.5.5|
|License:|Licensed|
|Edition:|Official|
|Language:|[en]|

{:.h2_title}
## Included Models 
 - ner_clinical
 - assertion_dl
 - re_clinical
 - ner_posology