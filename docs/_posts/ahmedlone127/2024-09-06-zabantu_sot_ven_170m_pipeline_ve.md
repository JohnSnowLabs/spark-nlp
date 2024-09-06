---
layout: model
title: Venda zabantu_sot_ven_170m_pipeline pipeline XlmRoBertaEmbeddings from dsfsi
author: John Snow Labs
name: zabantu_sot_ven_170m_pipeline
date: 2024-09-06
tags: [ve, open_source, pipeline, onnx]
task: Embeddings
language: ve
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`zabantu_sot_ven_170m_pipeline` is a Venda model originally trained by dsfsi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/zabantu_sot_ven_170m_pipeline_ve_5.5.0_3.0_1725596424963.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/zabantu_sot_ven_170m_pipeline_ve_5.5.0_3.0_1725596424963.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("zabantu_sot_ven_170m_pipeline", lang = "ve")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("zabantu_sot_ven_170m_pipeline", lang = "ve")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|zabantu_sot_ven_170m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ve|
|Size:|646.5 MB|

## References

https://huggingface.co/dsfsi/zabantu-sot-ven-170m

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaEmbeddings