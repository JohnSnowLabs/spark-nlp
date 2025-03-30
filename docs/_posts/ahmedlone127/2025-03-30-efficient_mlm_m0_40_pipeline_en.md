---
layout: model
title: English efficient_mlm_m0_40_pipeline pipeline RoBertaEmbeddings from princeton-nlp
author: John Snow Labs
name: efficient_mlm_m0_40_pipeline
date: 2025-03-30
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`efficient_mlm_m0_40_pipeline` is a English model originally trained by princeton-nlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/efficient_mlm_m0_40_pipeline_en_5.5.1_3.0_1743369337527.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/efficient_mlm_m0_40_pipeline_en_5.5.1_3.0_1743369337527.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("efficient_mlm_m0_40_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("efficient_mlm_m0_40_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|efficient_mlm_m0_40_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|844.4 MB|

## References

https://huggingface.co/princeton-nlp/efficient_mlm_m0.40

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings