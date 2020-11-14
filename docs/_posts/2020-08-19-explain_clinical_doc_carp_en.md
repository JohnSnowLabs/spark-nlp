---
layout: model
title: Explain Clinical Doc CARP
author: John Snow Labs
name: explain_clinical_doc_carp
date: 2020-08-19
tags: [pipeline, en, licensed]
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

annotations =  carp_pipeline.fullAnnotate("""Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain""")[0]

annotations.keys()

```

```scala

val carp_pipeline = new PretrainedPipeline("explain_clinical_doc_carp","en","clinical/models")

val result = carp_pipeline.fullAnnotate("""Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain""")(0)

```
</div>

{:.h2_title}
## Results
This pretrained pipeline gives the result of `ner_clinical`, `re_clinical`, `ner_posology` and `assertion_dl` models. Here is the result of `ner_clinical` and `ner_posology`:
```bash
|      tokens  | clinical_ner_tags | posology_ner_tags | dependencies |
|--------------|-------------------|-------------------|--------------|
| gestational  | B-PROBLEM         | O                 | of           |
| diabetes     | I-PROBLEM         | O                 | mellitus     |
| mellitus     | I-PROBLEM         | O                 | gestational  |
| metformin    | B-TREATMENT       | B-Drug            | take         |
| 1000         | O                 | B-Strength        | metformin    |
| mg           | O                 | I-Strength        | 1000         |
| two          | O                 | B-Frequency       | times        |
| times        | O                 | I-Frequency       | mg           |
| a            | O                 | I-Frequency       | day          |
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
 
