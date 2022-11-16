---
layout: model
title: NER Model Finder
author: John Snow Labs
name: ner_model_finder
date: 2021-11-24
tags: [pretrainedpipeline, clinical, ner, en, licensed]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 3.3.2
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is trained with bert embeddings and can be used to find the most appropriate NER model given the entity name.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_model_finder_en_3.3.2_2.4_1637761259895.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
ner_pipeline = PretrainedPipeline("ner_model_finder", "en", "clinical/models")

result = ner_pipeline.annotate("medication")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val ner_pipeline = PretrainedPipeline("ner_model_finder","en","clinical/models")

val result = ner_pipeline.annotate("medication")
```
</div>

## Results

```bash
{'model_names': ["['ner_posology', 'ner_posology_large', 'ner_posology_small', 'ner_posology_greedy', 'ner_drugs_large',  'ner_posology_experimental', 'ner_drugs_greedy', 'ner_ade_clinical', 'ner_jsl_slim', 'ner_posology_healthcare', 'ner_ade_healthcare', 'jsl_ner_wip_modifier_clinical', 'ner_ade_clinical', 'ner_jsl_greedy', 'ner_risk_factors']"]}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_model_finder|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- SentenceEntityResolverModel
- Finisher