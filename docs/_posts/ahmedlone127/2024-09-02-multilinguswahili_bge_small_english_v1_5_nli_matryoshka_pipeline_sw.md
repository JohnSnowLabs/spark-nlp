---
layout: model
title: Swahili (macrolanguage) multilinguswahili_bge_small_english_v1_5_nli_matryoshka_pipeline pipeline BGEEmbeddings from sartifyllc
author: John Snow Labs
name: multilinguswahili_bge_small_english_v1_5_nli_matryoshka_pipeline
date: 2024-09-02
tags: [sw, open_source, pipeline, onnx]
task: Embeddings
language: sw
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multilinguswahili_bge_small_english_v1_5_nli_matryoshka_pipeline` is a Swahili (macrolanguage) model originally trained by sartifyllc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multilinguswahili_bge_small_english_v1_5_nli_matryoshka_pipeline_sw_5.5.0_3.0_1725263638853.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multilinguswahili_bge_small_english_v1_5_nli_matryoshka_pipeline_sw_5.5.0_3.0_1725263638853.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multilinguswahili_bge_small_english_v1_5_nli_matryoshka_pipeline", lang = "sw")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multilinguswahili_bge_small_english_v1_5_nli_matryoshka_pipeline", lang = "sw")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multilinguswahili_bge_small_english_v1_5_nli_matryoshka_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sw|
|Size:|122.9 MB|

## References

https://huggingface.co/sartifyllc/MultiLinguSwahili-bge-small-en-v1.5-nli-matryoshka

## Included Models

- DocumentAssembler
- BGEEmbeddings