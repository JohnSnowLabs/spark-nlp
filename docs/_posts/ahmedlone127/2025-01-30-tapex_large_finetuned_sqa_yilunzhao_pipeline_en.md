---
layout: model
title: English tapex_large_finetuned_sqa_yilunzhao_pipeline pipeline BartTransformer from yilunzhao
author: John Snow Labs
name: tapex_large_finetuned_sqa_yilunzhao_pipeline
date: 2025-01-30
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

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tapex_large_finetuned_sqa_yilunzhao_pipeline` is a English model originally trained by yilunzhao.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tapex_large_finetuned_sqa_yilunzhao_pipeline_en_5.5.1_3.0_1738265647525.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tapex_large_finetuned_sqa_yilunzhao_pipeline_en_5.5.1_3.0_1738265647525.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tapex_large_finetuned_sqa_yilunzhao_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tapex_large_finetuned_sqa_yilunzhao_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tapex_large_finetuned_sqa_yilunzhao_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.9 GB|

## References

https://huggingface.co/yilunzhao/tapex-large-finetuned-sqa

## Included Models

- DocumentAssembler
- BartTransformer