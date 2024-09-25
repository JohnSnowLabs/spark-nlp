---
layout: model
title: English ibert_roberta_base_finetuned_imdb_pipeline pipeline RoBertaEmbeddings from elayat
author: John Snow Labs
name: ibert_roberta_base_finetuned_imdb_pipeline
date: 2024-09-20
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ibert_roberta_base_finetuned_imdb_pipeline` is a English model originally trained by elayat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ibert_roberta_base_finetuned_imdb_pipeline_en_5.5.0_3.0_1726815729901.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ibert_roberta_base_finetuned_imdb_pipeline_en_5.5.0_3.0_1726815729901.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ibert_roberta_base_finetuned_imdb_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ibert_roberta_base_finetuned_imdb_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ibert_roberta_base_finetuned_imdb_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.3 MB|

## References

https://huggingface.co/elayat/ibert-roberta-base-finetuned-imdb

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings