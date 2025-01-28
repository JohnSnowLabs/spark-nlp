---
layout: model
title: Czech mt5_base_multi_label_czech_iiib_pipeline pipeline T5Transformer from chi2024
author: John Snow Labs
name: mt5_base_multi_label_czech_iiib_pipeline
date: 2025-01-28
tags: [cs, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: cs
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_base_multi_label_czech_iiib_pipeline` is a Czech model originally trained by chi2024.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_base_multi_label_czech_iiib_pipeline_cs_5.5.1_3.0_1738074138263.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_base_multi_label_czech_iiib_pipeline_cs_5.5.1_3.0_1738074138263.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("mt5_base_multi_label_czech_iiib_pipeline", lang = "cs")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("mt5_base_multi_label_czech_iiib_pipeline", lang = "cs")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_base_multi_label_czech_iiib_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|cs|
|Size:|2.3 GB|

## References

References

https://huggingface.co/chi2024/mt5-base-multi-label-cs-iiib

## Included Models

- DocumentAssembler
- T5Transformer