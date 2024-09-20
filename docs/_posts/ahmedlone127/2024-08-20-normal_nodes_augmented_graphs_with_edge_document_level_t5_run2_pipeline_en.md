---
layout: model
title: English normal_nodes_augmented_graphs_with_edge_document_level_t5_run2_pipeline pipeline T5Transformer from sheoran95
author: John Snow Labs
name: normal_nodes_augmented_graphs_with_edge_document_level_t5_run2_pipeline
date: 2024-08-20
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`normal_nodes_augmented_graphs_with_edge_document_level_t5_run2_pipeline` is a English model originally trained by sheoran95.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/normal_nodes_augmented_graphs_with_edge_document_level_t5_run2_pipeline_en_5.4.2_3.0_1724195572248.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/normal_nodes_augmented_graphs_with_edge_document_level_t5_run2_pipeline_en_5.4.2_3.0_1724195572248.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("normal_nodes_augmented_graphs_with_edge_document_level_t5_run2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("normal_nodes_augmented_graphs_with_edge_document_level_t5_run2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|normal_nodes_augmented_graphs_with_edge_document_level_t5_run2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|313.6 MB|

## References

https://huggingface.co/sheoran95/normal_nodes_augmented_graphs_with_edge_document_level_T5_run2

## Included Models

- DocumentAssembler
- T5Transformer