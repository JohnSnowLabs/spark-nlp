---
layout: model
title: English weighted_cross_entropy_uniform_minority_weights_pipeline pipeline DistilBertForTokenClassification from pmpmp74832
author: John Snow Labs
name: weighted_cross_entropy_uniform_minority_weights_pipeline
date: 2025-01-23
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`weighted_cross_entropy_uniform_minority_weights_pipeline` is a English model originally trained by pmpmp74832.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/weighted_cross_entropy_uniform_minority_weights_pipeline_en_5.5.1_3.0_1737627256616.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/weighted_cross_entropy_uniform_minority_weights_pipeline_en_5.5.1_3.0_1737627256616.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("weighted_cross_entropy_uniform_minority_weights_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("weighted_cross_entropy_uniform_minority_weights_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|weighted_cross_entropy_uniform_minority_weights_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|329.4 MB|

## References

https://huggingface.co/pmpmp74832/weighted_cross_entropy_uniform_minority_weights

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification