---
layout: model
title: Hindi marianmt_hin_eng_czech_pipeline pipeline MarianTransformer from ar5entum
author: John Snow Labs
name: marianmt_hin_eng_czech_pipeline
date: 2024-09-02
tags: [hi, open_source, pipeline, onnx]
task: Translation
language: hi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marianmt_hin_eng_czech_pipeline` is a Hindi model originally trained by ar5entum.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marianmt_hin_eng_czech_pipeline_hi_5.5.0_3.0_1725295207563.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marianmt_hin_eng_czech_pipeline_hi_5.5.0_3.0_1725295207563.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("marianmt_hin_eng_czech_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("marianmt_hin_eng_czech_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marianmt_hin_eng_czech_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|533.1 MB|

## References

https://huggingface.co/ar5entum/marianMT_hin_eng_cs

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer