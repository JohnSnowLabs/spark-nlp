---
layout: model
title: English finetuned_caption_embedding_iccv2025submission_pipeline pipeline MPNetEmbeddings from iccv2025submission
author: John Snow Labs
name: finetuned_caption_embedding_iccv2025submission_pipeline
date: 2025-03-27
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_caption_embedding_iccv2025submission_pipeline` is a English model originally trained by iccv2025submission.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_caption_embedding_iccv2025submission_pipeline_en_5.5.1_3.0_1743117538082.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_caption_embedding_iccv2025submission_pipeline_en_5.5.1_3.0_1743117538082.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_caption_embedding_iccv2025submission_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_caption_embedding_iccv2025submission_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_caption_embedding_iccv2025submission_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.9 MB|

## References

https://huggingface.co/iccv2025submission/finetuned-caption-embedding

## Included Models

- DocumentAssembler
- MPNetEmbeddings