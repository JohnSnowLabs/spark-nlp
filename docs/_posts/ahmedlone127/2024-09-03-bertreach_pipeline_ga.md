---
layout: model
title: Irish bertreach_pipeline pipeline RoBertaEmbeddings from jimregan
author: John Snow Labs
name: bertreach_pipeline
date: 2024-09-03
tags: [ga, open_source, pipeline, onnx]
task: Embeddings
language: ga
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertreach_pipeline` is a Irish model originally trained by jimregan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertreach_pipeline_ga_5.5.0_3.0_1725381840933.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertreach_pipeline_ga_5.5.0_3.0_1725381840933.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertreach_pipeline", lang = "ga")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertreach_pipeline", lang = "ga")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertreach_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ga|
|Size:|311.6 MB|

## References

https://huggingface.co/jimregan/BERTreach

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings