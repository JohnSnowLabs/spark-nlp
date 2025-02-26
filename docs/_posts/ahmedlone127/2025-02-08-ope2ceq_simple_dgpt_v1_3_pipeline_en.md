---
layout: model
title: English ope2ceq_simple_dgpt_v1_3_pipeline pipeline GPT2Transformer from RyotaroOKabe
author: John Snow Labs
name: ope2ceq_simple_dgpt_v1_3_pipeline
date: 2025-02-08
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ope2ceq_simple_dgpt_v1_3_pipeline` is a English model originally trained by RyotaroOKabe.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ope2ceq_simple_dgpt_v1_3_pipeline_en_5.5.1_3.0_1739025991956.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ope2ceq_simple_dgpt_v1_3_pipeline_en_5.5.1_3.0_1739025991956.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ope2ceq_simple_dgpt_v1_3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ope2ceq_simple_dgpt_v1_3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ope2ceq_simple_dgpt_v1_3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|308.4 MB|

## References

https://huggingface.co/RyotaroOKabe/ope2ceq_simple_dgpt_v1.3

## Included Models

- DocumentAssembler
- GPT2Transformer