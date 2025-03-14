---
layout: model
title: English baai_bge_large_english_v1_5_fine_tuned_fold2_20241115_191836_pipeline pipeline BGEEmbeddings from marumarukun
author: John Snow Labs
name: baai_bge_large_english_v1_5_fine_tuned_fold2_20241115_191836_pipeline
date: 2024-12-18
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`baai_bge_large_english_v1_5_fine_tuned_fold2_20241115_191836_pipeline` is a English model originally trained by marumarukun.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/baai_bge_large_english_v1_5_fine_tuned_fold2_20241115_191836_pipeline_en_5.5.1_3.0_1734562745937.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/baai_bge_large_english_v1_5_fine_tuned_fold2_20241115_191836_pipeline_en_5.5.1_3.0_1734562745937.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("baai_bge_large_english_v1_5_fine_tuned_fold2_20241115_191836_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("baai_bge_large_english_v1_5_fine_tuned_fold2_20241115_191836_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|baai_bge_large_english_v1_5_fine_tuned_fold2_20241115_191836_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/marumarukun/BAAI-bge-large-en-v1.5_fine_tuned_fold2_20241115_191836

## Included Models

- DocumentAssembler
- BGEEmbeddings