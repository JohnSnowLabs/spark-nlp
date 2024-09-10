---
layout: model
title: Catalan, Valencian julibert_pipeline pipeline RoBertaEmbeddings from softcatala
author: John Snow Labs
name: julibert_pipeline
date: 2024-09-07
tags: [ca, open_source, pipeline, onnx]
task: Embeddings
language: ca
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`julibert_pipeline` is a Catalan, Valencian model originally trained by softcatala.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/julibert_pipeline_ca_5.5.0_3.0_1725678554029.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/julibert_pipeline_ca_5.5.0_3.0_1725678554029.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("julibert_pipeline", lang = "ca")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("julibert_pipeline", lang = "ca")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|julibert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ca|
|Size:|465.8 MB|

## References

https://huggingface.co/softcatala/julibert

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings