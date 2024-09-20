---
layout: model
title: English false_large_pmi_para0_sent1_span2_ittrue_sargmax_rrfalse_8_1024_0_3_best_pipeline pipeline T5Transformer from tau
author: John Snow Labs
name: false_large_pmi_para0_sent1_span2_ittrue_sargmax_rrfalse_8_1024_0_3_best_pipeline
date: 2024-08-09
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`false_large_pmi_para0_sent1_span2_ittrue_sargmax_rrfalse_8_1024_0_3_best_pipeline` is a English model originally trained by tau.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/false_large_pmi_para0_sent1_span2_ittrue_sargmax_rrfalse_8_1024_0_3_best_pipeline_en_5.4.2_3.0_1723241652553.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/false_large_pmi_para0_sent1_span2_ittrue_sargmax_rrfalse_8_1024_0_3_best_pipeline_en_5.4.2_3.0_1723241652553.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("false_large_pmi_para0_sent1_span2_ittrue_sargmax_rrfalse_8_1024_0_3_best_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("false_large_pmi_para0_sent1_span2_ittrue_sargmax_rrfalse_8_1024_0_3_best_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|false_large_pmi_para0_sent1_span2_ittrue_sargmax_rrfalse_8_1024_0_3_best_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.6 GB|

## References

https://huggingface.co/tau/False_large_pmi_para0_sent1_span2_itTrue_sargmax_rrFalse_8_1024_0.3_best

## Included Models

- DocumentAssembler
- T5Transformer