---
layout: model
title: English nusiss_election_project_entity_model_distilbert_base_cased_pipeline pipeline DistilBertForSequenceClassification from destonedbob
author: John Snow Labs
name: nusiss_election_project_entity_model_distilbert_base_cased_pipeline
date: 2024-11-11
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nusiss_election_project_entity_model_distilbert_base_cased_pipeline` is a English model originally trained by destonedbob.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nusiss_election_project_entity_model_distilbert_base_cased_pipeline_en_5.5.1_3.0_1731331616104.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nusiss_election_project_entity_model_distilbert_base_cased_pipeline_en_5.5.1_3.0_1731331616104.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nusiss_election_project_entity_model_distilbert_base_cased_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nusiss_election_project_entity_model_distilbert_base_cased_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nusiss_election_project_entity_model_distilbert_base_cased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|246.0 MB|

## References

https://huggingface.co/destonedbob/nusiss-election-project-entity-model-distilbert-base-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification