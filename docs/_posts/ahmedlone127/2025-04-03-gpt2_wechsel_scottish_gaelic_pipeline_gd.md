---
layout: model
title: Gaelic, Scottish Gaelic gpt2_wechsel_scottish_gaelic_pipeline pipeline GPT2Transformer from benjamin
author: John Snow Labs
name: gpt2_wechsel_scottish_gaelic_pipeline
date: 2025-04-03
tags: [gd, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: gd
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_wechsel_scottish_gaelic_pipeline` is a Gaelic, Scottish Gaelic model originally trained by benjamin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_wechsel_scottish_gaelic_pipeline_gd_5.5.1_3.0_1743664203331.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_wechsel_scottish_gaelic_pipeline_gd_5.5.1_3.0_1743664203331.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_wechsel_scottish_gaelic_pipeline", lang = "gd")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_wechsel_scottish_gaelic_pipeline", lang = "gd")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_wechsel_scottish_gaelic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|gd|
|Size:|468.1 MB|

## References

https://huggingface.co/benjamin/gpt2-wechsel-scottish-gaelic

## Included Models

- DocumentAssembler
- GPT2Transformer