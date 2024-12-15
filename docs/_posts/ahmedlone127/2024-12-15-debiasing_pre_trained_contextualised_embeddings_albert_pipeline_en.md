---
layout: model
title: English debiasing_pre_trained_contextualised_embeddings_albert_pipeline pipeline AlbertEmbeddings from Daniel-Saeedi
author: John Snow Labs
name: debiasing_pre_trained_contextualised_embeddings_albert_pipeline
date: 2024-12-15
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`debiasing_pre_trained_contextualised_embeddings_albert_pipeline` is a English model originally trained by Daniel-Saeedi.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/debiasing_pre_trained_contextualised_embeddings_albert_pipeline_en_5.5.1_3.0_1734245780576.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/debiasing_pre_trained_contextualised_embeddings_albert_pipeline_en_5.5.1_3.0_1734245780576.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("debiasing_pre_trained_contextualised_embeddings_albert_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("debiasing_pre_trained_contextualised_embeddings_albert_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|debiasing_pre_trained_contextualised_embeddings_albert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|42.0 MB|

## References

References

https://huggingface.co/Daniel-Saeedi/debiasing_pre-trained_contextualised_embeddings_albert

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertEmbeddings