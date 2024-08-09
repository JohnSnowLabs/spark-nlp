---
layout: model
title: Mongolian mongolian_g2p_t5_small_pipeline pipeline T5Transformer from bilguun
author: John Snow Labs
name: mongolian_g2p_t5_small_pipeline
date: 2024-08-01
tags: [mn, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: mn
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mongolian_g2p_t5_small_pipeline` is a Mongolian model originally trained by bilguun.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mongolian_g2p_t5_small_pipeline_mn_5.4.2_3.0_1722547855503.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mongolian_g2p_t5_small_pipeline_mn_5.4.2_3.0_1722547855503.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mongolian_g2p_t5_small_pipeline", lang = "mn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mongolian_g2p_t5_small_pipeline", lang = "mn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mongolian_g2p_t5_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|mn|
|Size:|188.9 MB|

## References

https://huggingface.co/bilguun/mn-g2p-t5-small

## Included Models

- DocumentAssembler
- T5Transformer