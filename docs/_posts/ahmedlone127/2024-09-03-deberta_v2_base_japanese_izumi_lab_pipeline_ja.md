---
layout: model
title: Japanese deberta_v2_base_japanese_izumi_lab_pipeline pipeline DeBertaEmbeddings from izumi-lab
author: John Snow Labs
name: deberta_v2_base_japanese_izumi_lab_pipeline
date: 2024-09-03
tags: [ja, open_source, pipeline, onnx]
task: Embeddings
language: ja
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_v2_base_japanese_izumi_lab_pipeline` is a Japanese model originally trained by izumi-lab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v2_base_japanese_izumi_lab_pipeline_ja_5.5.0_3.0_1725331779251.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_v2_base_japanese_izumi_lab_pipeline_ja_5.5.0_3.0_1725331779251.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deberta_v2_base_japanese_izumi_lab_pipeline", lang = "ja")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deberta_v2_base_japanese_izumi_lab_pipeline", lang = "ja")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v2_base_japanese_izumi_lab_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ja|
|Size:|263.1 MB|

## References

https://huggingface.co/izumi-lab/deberta-v2-base-japanese

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaEmbeddings