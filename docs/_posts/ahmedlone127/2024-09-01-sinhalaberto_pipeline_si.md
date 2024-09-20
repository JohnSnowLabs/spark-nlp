---
layout: model
title: Sinhala, Sinhalese sinhalaberto_pipeline pipeline RoBertaEmbeddings from keshan
author: John Snow Labs
name: sinhalaberto_pipeline
date: 2024-09-01
tags: [si, open_source, pipeline, onnx]
task: Embeddings
language: si
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sinhalaberto_pipeline` is a Sinhala, Sinhalese model originally trained by keshan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sinhalaberto_pipeline_si_5.4.2_3.0_1725165084460.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sinhalaberto_pipeline_si_5.4.2_3.0_1725165084460.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sinhalaberto_pipeline", lang = "si")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sinhalaberto_pipeline", lang = "si")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sinhalaberto_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|si|
|Size:|311.7 MB|

## References

https://huggingface.co/keshan/SinhalaBERTo

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings