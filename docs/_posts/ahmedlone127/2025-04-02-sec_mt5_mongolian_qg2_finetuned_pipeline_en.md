---
layout: model
title: English sec_mt5_mongolian_qg2_finetuned_pipeline pipeline T5Transformer from Alt4nsuh
author: John Snow Labs
name: sec_mt5_mongolian_qg2_finetuned_pipeline
date: 2025-04-02
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sec_mt5_mongolian_qg2_finetuned_pipeline` is a English model originally trained by Alt4nsuh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sec_mt5_mongolian_qg2_finetuned_pipeline_en_5.5.1_3.0_1743556858389.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sec_mt5_mongolian_qg2_finetuned_pipeline_en_5.5.1_3.0_1743556858389.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sec_mt5_mongolian_qg2_finetuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sec_mt5_mongolian_qg2_finetuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sec_mt5_mongolian_qg2_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/Alt4nsuh/sec-mt5-mn-qg2-finetuned

## Included Models

- DocumentAssembler
- T5Transformer