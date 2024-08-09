---
layout: model
title: Turkish turkish_t5_pipeline pipeline T5Transformer from emirhangazi77
author: John Snow Labs
name: turkish_t5_pipeline
date: 2024-08-06
tags: [tr, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: tr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`turkish_t5_pipeline` is a Turkish model originally trained by emirhangazi77.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/turkish_t5_pipeline_tr_5.4.2_3.0_1722907847154.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/turkish_t5_pipeline_tr_5.4.2_3.0_1722907847154.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("turkish_t5_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("turkish_t5_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|turkish_t5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|655.1 MB|

## References

https://huggingface.co/emirhangazi77/Turkish-T5

## Included Models

- DocumentAssembler
- T5Transformer