---
layout: model
title: Lithuanian litberta_uncased_pipeline pipeline RoBertaEmbeddings from jkeruotis
author: John Snow Labs
name: litberta_uncased_pipeline
date: 2024-09-05
tags: [lt, open_source, pipeline, onnx]
task: Embeddings
language: lt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`litberta_uncased_pipeline` is a Lithuanian model originally trained by jkeruotis.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/litberta_uncased_pipeline_lt_5.5.0_3.0_1725578703248.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/litberta_uncased_pipeline_lt_5.5.0_3.0_1725578703248.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("litberta_uncased_pipeline", lang = "lt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("litberta_uncased_pipeline", lang = "lt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|litberta_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|lt|
|Size:|689.2 MB|

## References

https://huggingface.co/jkeruotis/LitBERTa-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings