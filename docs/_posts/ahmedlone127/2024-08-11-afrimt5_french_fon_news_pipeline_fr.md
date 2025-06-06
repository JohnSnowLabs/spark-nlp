---
layout: model
title: French afrimt5_french_fon_news_pipeline pipeline T5Transformer from masakhane
author: John Snow Labs
name: afrimt5_french_fon_news_pipeline
date: 2024-08-11
tags: [fr, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: fr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`afrimt5_french_fon_news_pipeline` is a French model originally trained by masakhane.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/afrimt5_french_fon_news_pipeline_fr_5.4.2_3.0_1723383478041.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/afrimt5_french_fon_news_pipeline_fr_5.4.2_3.0_1723383478041.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("afrimt5_french_fon_news_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("afrimt5_french_fon_news_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|afrimt5_french_fon_news_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|2.8 GB|

## References

https://huggingface.co/masakhane/afrimt5_fr_fon_news

## Included Models

- DocumentAssembler
- T5Transformer