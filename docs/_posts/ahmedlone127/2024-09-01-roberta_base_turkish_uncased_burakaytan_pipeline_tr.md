---
layout: model
title: Turkish roberta_base_turkish_uncased_burakaytan_pipeline pipeline RoBertaEmbeddings from burakaytan
author: John Snow Labs
name: roberta_base_turkish_uncased_burakaytan_pipeline
date: 2024-09-01
tags: [tr, open_source, pipeline, onnx]
task: Embeddings
language: tr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_turkish_uncased_burakaytan_pipeline` is a Turkish model originally trained by burakaytan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_turkish_uncased_burakaytan_pipeline_tr_5.4.2_3.0_1725164971178.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_turkish_uncased_burakaytan_pipeline_tr_5.4.2_3.0_1725164971178.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_turkish_uncased_burakaytan_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_turkish_uncased_burakaytan_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_turkish_uncased_burakaytan_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|461.5 MB|

## References

https://huggingface.co/burakaytan/roberta-base-turkish-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings