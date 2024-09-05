---
layout: model
title: Slovak slovakbert_pipeline pipeline RoBertaEmbeddings from gerulata
author: John Snow Labs
name: slovakbert_pipeline
date: 2024-09-03
tags: [sk, open_source, pipeline, onnx]
task: Embeddings
language: sk
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`slovakbert_pipeline` is a Slovak model originally trained by gerulata.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/slovakbert_pipeline_sk_5.5.0_3.0_1725375748181.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/slovakbert_pipeline_sk_5.5.0_3.0_1725375748181.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("slovakbert_pipeline", lang = "sk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("slovakbert_pipeline", lang = "sk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|slovakbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sk|
|Size:|296.1 MB|

## References

https://huggingface.co/gerulata/slovakbert

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings