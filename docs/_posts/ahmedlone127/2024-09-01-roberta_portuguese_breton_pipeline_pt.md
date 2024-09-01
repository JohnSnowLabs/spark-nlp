---
layout: model
title: Portuguese roberta_portuguese_breton_pipeline pipeline RoBertaEmbeddings from josu
author: John Snow Labs
name: roberta_portuguese_breton_pipeline
date: 2024-09-01
tags: [pt, open_source, pipeline, onnx]
task: Embeddings
language: pt
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_portuguese_breton_pipeline` is a Portuguese model originally trained by josu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_portuguese_breton_pipeline_pt_5.4.2_3.0_1725164678943.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_portuguese_breton_pipeline_pt_5.4.2_3.0_1725164678943.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_portuguese_breton_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_portuguese_breton_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_portuguese_breton_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|306.7 MB|

## References

https://huggingface.co/josu/roberta-pt-br

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings