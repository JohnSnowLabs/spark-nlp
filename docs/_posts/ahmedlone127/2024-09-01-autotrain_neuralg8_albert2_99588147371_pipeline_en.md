---
layout: model
title: English autotrain_neuralg8_albert2_99588147371_pipeline pipeline AlbertForQuestionAnswering from Samis922
author: John Snow Labs
name: autotrain_neuralg8_albert2_99588147371_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_neuralg8_albert2_99588147371_pipeline` is a English model originally trained by Samis922.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_neuralg8_albert2_99588147371_pipeline_en_5.4.2_3.0_1725193489272.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_neuralg8_albert2_99588147371_pipeline_en_5.4.2_3.0_1725193489272.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_neuralg8_albert2_99588147371_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_neuralg8_albert2_99588147371_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_neuralg8_albert2_99588147371_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|42.0 MB|

## References

https://huggingface.co/Samis922/autotrain-neuralg8_albert2-99588147371

## Included Models

- MultiDocumentAssembler
- AlbertForQuestionAnswering