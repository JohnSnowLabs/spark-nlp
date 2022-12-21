---
layout: model
title: NER Model Finder
author: John Snow Labs
name: ner_model_finder
date: 2022-09-05
tags: [pretrainedpipeline, clinical, ner, en, licensed]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.1.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is trained with bert embeddings and can be used to find the most appropriate NER model given the entity name.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_MODEL_FINDER/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_model_finder_en_4.1.0_3.0_1662378666469.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_model_finder_en_4.1.0_3.0_1662378666469.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
{'model_names': ["['ner_posology_greedy', 'jsl_ner_wip_modifier_clinical', 'ner_posology_small', 'ner_jsl_greedy', 'ner_ade_clinical', 'ner_posology', 'ner_risk_factors', 'ner_ade_healthcare', 'ner_drugs_large', 'ner_jsl_slim', 'ner_posology_experimental', 'ner_posology_large', 'ner_posology_healthcare', 'ner_drugs_greedy', 'ner_pathogen']"]}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_model_finder|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.1.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|155.9 MB|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- SentenceEntityResolverModel
- Finisher