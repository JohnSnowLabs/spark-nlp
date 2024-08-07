---
layout: model
title: None english_tonga_tonga_islands_nepali_translate_pipeline pipeline T5Transformer from Hemg
author: John Snow Labs
name: english_tonga_tonga_islands_nepali_translate_pipeline
date: 2024-08-07
tags: [nan, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: nan
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`english_tonga_tonga_islands_nepali_translate_pipeline` is a None model originally trained by Hemg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/english_tonga_tonga_islands_nepali_translate_pipeline_nan_5.4.2_3.0_1723041964040.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/english_tonga_tonga_islands_nepali_translate_pipeline_nan_5.4.2_3.0_1723041964040.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("english_tonga_tonga_islands_nepali_translate_pipeline", lang = "nan")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("english_tonga_tonga_islands_nepali_translate_pipeline", lang = "nan")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|english_tonga_tonga_islands_nepali_translate_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|nan|
|Size:|324.0 MB|

## References

https://huggingface.co/Hemg/english-To-Nepali-TRanslate

## Included Models

- DocumentAssembler
- T5Transformer