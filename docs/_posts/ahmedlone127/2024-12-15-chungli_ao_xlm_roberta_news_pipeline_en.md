---
layout: model
title: English chungli_ao_xlm_roberta_news_pipeline pipeline XlmRoBertaEmbeddings from N1ch0
author: John Snow Labs
name: chungli_ao_xlm_roberta_news_pipeline
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

Pretrained XlmRoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`chungli_ao_xlm_roberta_news_pipeline` is a English model originally trained by N1ch0.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/chungli_ao_xlm_roberta_news_pipeline_en_5.5.1_3.0_1734229401421.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/chungli_ao_xlm_roberta_news_pipeline_en_5.5.1_3.0_1734229401421.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("chungli_ao_xlm_roberta_news_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("chungli_ao_xlm_roberta_news_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chungli_ao_xlm_roberta_news_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/N1ch0/chungli-ao-xlm-roberta-news

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaEmbeddings