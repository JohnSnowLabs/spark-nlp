---
layout: model
title: Czech mt5_base_multi_label_czech_iiib_02c_pipeline pipeline T5Transformer from chi2024
author: John Snow Labs
name: mt5_base_multi_label_czech_iiib_02c_pipeline
date: 2024-08-17
tags: [cs, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: cs
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_base_multi_label_czech_iiib_02c_pipeline` is a Czech model originally trained by chi2024.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_base_multi_label_czech_iiib_02c_pipeline_cs_5.4.2_3.0_1723859038042.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_base_multi_label_czech_iiib_02c_pipeline_cs_5.4.2_3.0_1723859038042.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_base_multi_label_czech_iiib_02c_pipeline", lang = "cs")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_base_multi_label_czech_iiib_02c_pipeline", lang = "cs")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_base_multi_label_czech_iiib_02c_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|cs|
|Size:|2.3 GB|

## References

https://huggingface.co/chi2024/mt5-base-multi-label-cs-iiib-02c

## Included Models

- DocumentAssembler
- T5Transformer