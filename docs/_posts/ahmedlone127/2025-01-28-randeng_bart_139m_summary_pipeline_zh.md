---
layout: model
title: Chinese randeng_bart_139m_summary_pipeline pipeline BartTransformer from IDEA-CCNL
author: John Snow Labs
name: randeng_bart_139m_summary_pipeline
date: 2025-01-28
tags: [zh, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: zh
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`randeng_bart_139m_summary_pipeline` is a Chinese model originally trained by IDEA-CCNL.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/randeng_bart_139m_summary_pipeline_zh_5.5.1_3.0_1738079780343.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/randeng_bart_139m_summary_pipeline_zh_5.5.1_3.0_1738079780343.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("randeng_bart_139m_summary_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("randeng_bart_139m_summary_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|randeng_bart_139m_summary_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|514.4 MB|

## References

https://huggingface.co/IDEA-CCNL/Randeng-BART-139M-SUMMARY

## Included Models

- DocumentAssembler
- BartTransformer