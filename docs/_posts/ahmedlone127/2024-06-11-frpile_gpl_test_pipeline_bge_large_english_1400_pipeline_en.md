---
layout: model
title: English frpile_gpl_test_pipeline_bge_large_english_1400_pipeline pipeline BGEEmbeddings from DragosGorduza
author: John Snow Labs
name: frpile_gpl_test_pipeline_bge_large_english_1400_pipeline
date: 2024-06-11
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`frpile_gpl_test_pipeline_bge_large_english_1400_pipeline` is a English model originally trained by DragosGorduza.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/frpile_gpl_test_pipeline_bge_large_english_1400_pipeline_en_5.4.0_3.0_1718064864537.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/frpile_gpl_test_pipeline_bge_large_english_1400_pipeline_en_5.4.0_3.0_1718064864537.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("frpile_gpl_test_pipeline_bge_large_english_1400_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("frpile_gpl_test_pipeline_bge_large_english_1400_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|frpile_gpl_test_pipeline_bge_large_english_1400_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/DragosGorduza/FRPile_GPL_test_pipeline_bge-large-en_1400

## Included Models

- DocumentAssembler
- BGEEmbeddings