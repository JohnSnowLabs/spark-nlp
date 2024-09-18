---
layout: model
title: Slovak fernet_news_slovak_pipeline pipeline RoBertaEmbeddings from fav-kky
author: John Snow Labs
name: fernet_news_slovak_pipeline
date: 2024-09-15
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fernet_news_slovak_pipeline` is a Slovak model originally trained by fav-kky.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fernet_news_slovak_pipeline_sk_5.5.0_3.0_1726383976836.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fernet_news_slovak_pipeline_sk_5.5.0_3.0_1726383976836.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fernet_news_slovak_pipeline", lang = "sk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fernet_news_slovak_pipeline", lang = "sk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fernet_news_slovak_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sk|
|Size:|464.6 MB|

## References

https://huggingface.co/fav-kky/FERNET-News_sk

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings