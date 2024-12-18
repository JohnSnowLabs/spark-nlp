---
layout: model
title: German germanfinbert_sardinian_pipeline pipeline BertEmbeddings from scherrmann
author: John Snow Labs
name: germanfinbert_sardinian_pipeline
date: 2024-12-17
tags: [de, open_source, pipeline, onnx]
task: Embeddings
language: de
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`germanfinbert_sardinian_pipeline` is a German model originally trained by scherrmann.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/germanfinbert_sardinian_pipeline_de_5.5.1_3.0_1734415729073.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/germanfinbert_sardinian_pipeline_de_5.5.1_3.0_1734415729073.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("germanfinbert_sardinian_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("germanfinbert_sardinian_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|germanfinbert_sardinian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|405.9 MB|

## References

https://huggingface.co/scherrmann/GermanFinBert_SC

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings