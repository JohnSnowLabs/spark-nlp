---
layout: model
title: Dutch, Flemish robbert_2022_dutch_base_pipeline pipeline RoBertaEmbeddings from DTAI-KULeuven
author: John Snow Labs
name: robbert_2022_dutch_base_pipeline
date: 2024-09-01
tags: [nl, open_source, pipeline, onnx]
task: Embeddings
language: nl
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`robbert_2022_dutch_base_pipeline` is a Dutch, Flemish model originally trained by DTAI-KULeuven.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/robbert_2022_dutch_base_pipeline_nl_5.4.2_3.0_1725165132015.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/robbert_2022_dutch_base_pipeline_nl_5.4.2_3.0_1725165132015.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("robbert_2022_dutch_base_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("robbert_2022_dutch_base_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robbert_2022_dutch_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|443.8 MB|

## References

https://huggingface.co/DTAI-KULeuven/robbert-2022-dutch-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings