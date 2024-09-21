---
layout: model
title: Hindi marianmt_bislama_dev_rom_tagalog_pipeline pipeline MarianTransformer from ar5entum
author: John Snow Labs
name: marianmt_bislama_dev_rom_tagalog_pipeline
date: 2024-09-17
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marianmt_bislama_dev_rom_tagalog_pipeline` is a Hindi model originally trained by ar5entum.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marianmt_bislama_dev_rom_tagalog_pipeline_hi_5.5.0_3.0_1726598996827.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marianmt_bislama_dev_rom_tagalog_pipeline_hi_5.5.0_3.0_1726598996827.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("marianmt_bislama_dev_rom_tagalog_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("marianmt_bislama_dev_rom_tagalog_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marianmt_bislama_dev_rom_tagalog_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|519.0 MB|

## References

https://huggingface.co/ar5entum/marianMT_bi_dev_rom_tl

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer