---
layout: model
title: Russian df_lik_n_malagasy_221_pipeline pipeline T5Transformer from uaritm
author: John Snow Labs
name: df_lik_n_malagasy_221_pipeline
date: 2024-08-03
tags: [ru, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ru
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`df_lik_n_malagasy_221_pipeline` is a Russian model originally trained by uaritm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/df_lik_n_malagasy_221_pipeline_ru_5.4.2_3.0_1722714391296.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/df_lik_n_malagasy_221_pipeline_ru_5.4.2_3.0_1722714391296.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("df_lik_n_malagasy_221_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("df_lik_n_malagasy_221_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|df_lik_n_malagasy_221_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|997.8 MB|

## References

https://huggingface.co/uaritm/df_lik_n_mg_221

## Included Models

- DocumentAssembler
- T5Transformer