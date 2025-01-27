---
layout: model
title: Slovak slovak_bpe_blm_pipeline pipeline RoBertaEmbeddings from daviddrzik
author: John Snow Labs
name: slovak_bpe_blm_pipeline
date: 2025-01-26
tags: [sk, open_source, pipeline, onnx]
task: Embeddings
language: sk
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`slovak_bpe_blm_pipeline` is a Slovak model originally trained by daviddrzik.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/slovak_bpe_blm_pipeline_sk_5.5.1_3.0_1737906668267.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/slovak_bpe_blm_pipeline_sk_5.5.1_3.0_1737906668267.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("slovak_bpe_blm_pipeline", lang = "sk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("slovak_bpe_blm_pipeline", lang = "sk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|slovak_bpe_blm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|sk|
|Size:|219.6 MB|

## References

https://huggingface.co/daviddrzik/SK_BPE_BLM

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings