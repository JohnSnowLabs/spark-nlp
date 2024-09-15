---
layout: model
title: Basque robasquerta_pipeline pipeline RoBertaEmbeddings from mrm8488
author: John Snow Labs
name: robasquerta_pipeline
date: 2024-09-07
tags: [eu, open_source, pipeline, onnx]
task: Embeddings
language: eu
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`robasquerta_pipeline` is a Basque model originally trained by mrm8488.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/robasquerta_pipeline_eu_5.5.0_3.0_1725672728216.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/robasquerta_pipeline_eu_5.5.0_3.0_1725672728216.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("robasquerta_pipeline", lang = "eu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("robasquerta_pipeline", lang = "eu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robasquerta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|eu|
|Size:|310.5 MB|

## References

https://huggingface.co/mrm8488/RoBasquERTa

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings