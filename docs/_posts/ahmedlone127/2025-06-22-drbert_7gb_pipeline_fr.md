---
layout: model
title: French drbert_7gb_pipeline pipeline CamemBertEmbeddings from Dr-BERT
author: John Snow Labs
name: drbert_7gb_pipeline
date: 2025-06-22
tags: [fr, open_source, pipeline, onnx]
task: Embeddings
language: fr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`drbert_7gb_pipeline` is a French model originally trained by Dr-BERT.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/drbert_7gb_pipeline_fr_5.5.1_3.0_1750618327381.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/drbert_7gb_pipeline_fr_5.5.1_3.0_1750618327381.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("drbert_7gb_pipeline", lang = "fr")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("drbert_7gb_pipeline", lang = "fr")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|drbert_7gb_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|412.6 MB|

## References

References

https://huggingface.co/Dr-BERT/DrBERT-7GB

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertEmbeddings