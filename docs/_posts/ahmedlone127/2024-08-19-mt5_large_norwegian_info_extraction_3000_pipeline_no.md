---
layout: model
title: Norwegian mt5_large_norwegian_info_extraction_3000_pipeline pipeline T5Transformer from norkart
author: John Snow Labs
name: mt5_large_norwegian_info_extraction_3000_pipeline
date: 2024-08-19
tags: ["no", open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: "no"
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_large_norwegian_info_extraction_3000_pipeline` is a Norwegian model originally trained by norkart.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_large_norwegian_info_extraction_3000_pipeline_no_5.4.2_3.0_1724063800531.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_large_norwegian_info_extraction_3000_pipeline_no_5.4.2_3.0_1724063800531.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_large_norwegian_info_extraction_3000_pipeline", lang = "no")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_large_norwegian_info_extraction_3000_pipeline", lang = "no")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_large_norwegian_info_extraction_3000_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|no|
|Size:|3.0 GB|

## References

https://huggingface.co/norkart/mt5-large-no-info-extraction-3000

## Included Models

- DocumentAssembler
- T5Transformer