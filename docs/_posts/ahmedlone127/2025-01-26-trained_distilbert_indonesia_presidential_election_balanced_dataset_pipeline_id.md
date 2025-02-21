---
layout: model
title: Indonesian trained_distilbert_indonesia_presidential_election_balanced_dataset_pipeline pipeline DistilBertForSequenceClassification from Rendika
author: John Snow Labs
name: trained_distilbert_indonesia_presidential_election_balanced_dataset_pipeline
date: 2025-01-26
tags: [id, open_source, pipeline, onnx]
task: Text Classification
language: id
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`trained_distilbert_indonesia_presidential_election_balanced_dataset_pipeline` is a Indonesian model originally trained by Rendika.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/trained_distilbert_indonesia_presidential_election_balanced_dataset_pipeline_id_5.5.1_3.0_1737873881596.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/trained_distilbert_indonesia_presidential_election_balanced_dataset_pipeline_id_5.5.1_3.0_1737873881596.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("trained_distilbert_indonesia_presidential_election_balanced_dataset_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("trained_distilbert_indonesia_presidential_election_balanced_dataset_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|trained_distilbert_indonesia_presidential_election_balanced_dataset_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|255.3 MB|

## References

https://huggingface.co/Rendika/Trained-DistilBERT-Indonesia-Presidential-Election-Balanced-Dataset

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification