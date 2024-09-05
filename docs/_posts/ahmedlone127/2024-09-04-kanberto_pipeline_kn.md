---
layout: model
title: Kannada kanberto_pipeline pipeline RoBertaEmbeddings from Naveen-k
author: John Snow Labs
name: kanberto_pipeline
date: 2024-09-04
tags: [kn, open_source, pipeline, onnx]
task: Embeddings
language: kn
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kanberto_pipeline` is a Kannada model originally trained by Naveen-k.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kanberto_pipeline_kn_5.5.0_3.0_1725412329196.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kanberto_pipeline_kn_5.5.0_3.0_1725412329196.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kanberto_pipeline", lang = "kn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kanberto_pipeline", lang = "kn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kanberto_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|kn|
|Size:|311.8 MB|

## References

https://huggingface.co/Naveen-k/KanBERTo

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings