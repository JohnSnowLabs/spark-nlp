---
layout: model
title: Macedonian macedonian_roberta_base_pipeline pipeline RoBertaEmbeddings from macedonizer
author: John Snow Labs
name: macedonian_roberta_base_pipeline
date: 2024-09-07
tags: [mk, open_source, pipeline, onnx]
task: Embeddings
language: mk
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`macedonian_roberta_base_pipeline` is a Macedonian model originally trained by macedonizer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/macedonian_roberta_base_pipeline_mk_5.5.0_3.0_1725678768825.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/macedonian_roberta_base_pipeline_mk_5.5.0_3.0_1725678768825.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("macedonian_roberta_base_pipeline", lang = "mk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("macedonian_roberta_base_pipeline", lang = "mk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|macedonian_roberta_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|mk|
|Size:|311.9 MB|

## References

https://huggingface.co/macedonizer/mk-roberta-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings