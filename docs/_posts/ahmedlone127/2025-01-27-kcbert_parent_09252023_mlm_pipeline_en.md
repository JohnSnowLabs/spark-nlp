---
layout: model
title: English kcbert_parent_09252023_mlm_pipeline pipeline BertEmbeddings from JuneKo
author: John Snow Labs
name: kcbert_parent_09252023_mlm_pipeline
date: 2025-01-27
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kcbert_parent_09252023_mlm_pipeline` is a English model originally trained by JuneKo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kcbert_parent_09252023_mlm_pipeline_en_5.5.1_3.0_1737984893659.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kcbert_parent_09252023_mlm_pipeline_en_5.5.1_3.0_1737984893659.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kcbert_parent_09252023_mlm_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kcbert_parent_09252023_mlm_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kcbert_parent_09252023_mlm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.2 MB|

## References

https://huggingface.co/JuneKo/kcBERT_parent_09252023_MLM

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings