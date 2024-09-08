---
layout: model
title: English finetuned_opusmt_english_tonga_tonga_islands_tamil_pipeline pipeline MarianTransformer from Varsha00
author: John Snow Labs
name: finetuned_opusmt_english_tonga_tonga_islands_tamil_pipeline
date: 2024-09-05
tags: [en, open_source, pipeline, onnx]
task: Translation
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_opusmt_english_tonga_tonga_islands_tamil_pipeline` is a English model originally trained by Varsha00.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_opusmt_english_tonga_tonga_islands_tamil_pipeline_en_5.5.0_3.0_1725545292121.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_opusmt_english_tonga_tonga_islands_tamil_pipeline_en_5.5.0_3.0_1725545292121.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_opusmt_english_tonga_tonga_islands_tamil_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_opusmt_english_tonga_tonga_islands_tamil_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_opusmt_english_tonga_tonga_islands_tamil_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|530.7 MB|

## References

https://huggingface.co/Varsha00/finetuned-opusmt-en-to-ta

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer