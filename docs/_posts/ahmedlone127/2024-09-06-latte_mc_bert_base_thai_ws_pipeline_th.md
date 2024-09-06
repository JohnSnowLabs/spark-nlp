---
layout: model
title: Thai latte_mc_bert_base_thai_ws_pipeline pipeline BertForTokenClassification from yacht
author: John Snow Labs
name: latte_mc_bert_base_thai_ws_pipeline
date: 2024-09-06
tags: [th, open_source, pipeline, onnx]
task: Named Entity Recognition
language: th
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`latte_mc_bert_base_thai_ws_pipeline` is a Thai model originally trained by yacht.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/latte_mc_bert_base_thai_ws_pipeline_th_5.5.0_3.0_1725634556869.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/latte_mc_bert_base_thai_ws_pipeline_th_5.5.0_3.0_1725634556869.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("latte_mc_bert_base_thai_ws_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("latte_mc_bert_base_thai_ws_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|latte_mc_bert_base_thai_ws_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|1.1 GB|

## References

https://huggingface.co/yacht/latte-mc-bert-base-thai-ws

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification