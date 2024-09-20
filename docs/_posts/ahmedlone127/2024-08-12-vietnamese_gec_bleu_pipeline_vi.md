---
layout: model
title: Vietnamese vietnamese_gec_bleu_pipeline pipeline T5Transformer from Huyen2310
author: John Snow Labs
name: vietnamese_gec_bleu_pipeline
date: 2024-08-12
tags: [vi, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: vi
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vietnamese_gec_bleu_pipeline` is a Vietnamese model originally trained by Huyen2310.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vietnamese_gec_bleu_pipeline_vi_5.4.2_3.0_1723444602180.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vietnamese_gec_bleu_pipeline_vi_5.4.2_3.0_1723444602180.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vietnamese_gec_bleu_pipeline", lang = "vi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vietnamese_gec_bleu_pipeline", lang = "vi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vietnamese_gec_bleu_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|vi|
|Size:|1.0 GB|

## References

https://huggingface.co/Huyen2310/Vi-gec-bleu

## Included Models

- DocumentAssembler
- T5Transformer