---
layout: model
title: English roberta_large_unlabeled_gab_reddit_semeval2023_task10_57000sample_pipeline pipeline RoBertaEmbeddings from HPL
author: John Snow Labs
name: roberta_large_unlabeled_gab_reddit_semeval2023_task10_57000sample_pipeline
date: 2024-09-17
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_large_unlabeled_gab_reddit_semeval2023_task10_57000sample_pipeline` is a English model originally trained by HPL.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_large_unlabeled_gab_reddit_semeval2023_task10_57000sample_pipeline_en_5.5.0_3.0_1726595262526.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_large_unlabeled_gab_reddit_semeval2023_task10_57000sample_pipeline_en_5.5.0_3.0_1726595262526.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_large_unlabeled_gab_reddit_semeval2023_task10_57000sample_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_large_unlabeled_gab_reddit_semeval2023_task10_57000sample_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_large_unlabeled_gab_reddit_semeval2023_task10_57000sample_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/HPL/roberta-large-unlabeled-gab-reddit-semeval2023-task10-57000sample

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings