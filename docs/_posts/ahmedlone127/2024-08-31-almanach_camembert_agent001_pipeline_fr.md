---
layout: model
title: French almanach_camembert_agent001_pipeline pipeline CamemBertEmbeddings from MisterAI
author: John Snow Labs
name: almanach_camembert_agent001_pipeline
date: 2024-08-31
tags: [fr, open_source, pipeline, onnx]
task: Embeddings
language: fr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`almanach_camembert_agent001_pipeline` is a French model originally trained by MisterAI.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/almanach_camembert_agent001_pipeline_fr_5.4.2_3.0_1725129688522.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/almanach_camembert_agent001_pipeline_fr_5.4.2_3.0_1725129688522.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("almanach_camembert_agent001_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("almanach_camembert_agent001_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|almanach_camembert_agent001_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|264.0 MB|

## References

https://huggingface.co/MisterAI/ALMANACH_CamemBERT_Agent001

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertEmbeddings