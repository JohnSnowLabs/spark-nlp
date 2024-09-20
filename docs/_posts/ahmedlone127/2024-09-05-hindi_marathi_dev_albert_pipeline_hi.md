---
layout: model
title: Hindi hindi_marathi_dev_albert_pipeline pipeline AlbertEmbeddings from l3cube-pune
author: John Snow Labs
name: hindi_marathi_dev_albert_pipeline
date: 2024-09-05
tags: [hi, open_source, pipeline, onnx]
task: Embeddings
language: hi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hindi_marathi_dev_albert_pipeline` is a Hindi model originally trained by l3cube-pune.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hindi_marathi_dev_albert_pipeline_hi_5.5.0_3.0_1725568447098.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hindi_marathi_dev_albert_pipeline_hi_5.5.0_3.0_1725568447098.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hindi_marathi_dev_albert_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hindi_marathi_dev_albert_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hindi_marathi_dev_albert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|125.3 MB|

## References

https://huggingface.co/l3cube-pune/hindi-marathi-dev-albert

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertEmbeddings