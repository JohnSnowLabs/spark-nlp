---
layout: model
title: English neuronale_crew_a6_pipeline pipeline DistilBertForSequenceClassification from ninjeanne-hka
author: John Snow Labs
name: neuronale_crew_a6_pipeline
date: 2024-09-19
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`neuronale_crew_a6_pipeline` is a English model originally trained by ninjeanne-hka.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/neuronale_crew_a6_pipeline_en_5.5.0_3.0_1726743552974.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/neuronale_crew_a6_pipeline_en_5.5.0_3.0_1726743552974.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("neuronale_crew_a6_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("neuronale_crew_a6_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|neuronale_crew_a6_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/ninjeanne-hka/neuronale_crew_a6

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification