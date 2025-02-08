---
layout: model
title: Multilingual bert_base_multilingual_uncased_finetuned_wikipedia_pipeline pipeline BertEmbeddings from rman-rahimi-29
author: John Snow Labs
name: bert_base_multilingual_uncased_finetuned_wikipedia_pipeline
date: 2025-02-05
tags: [xx, open_source, pipeline, onnx]
task: Embeddings
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_multilingual_uncased_finetuned_wikipedia_pipeline` is a Multilingual model originally trained by rman-rahimi-29.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_multilingual_uncased_finetuned_wikipedia_pipeline_xx_5.5.1_3.0_1738785212416.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_multilingual_uncased_finetuned_wikipedia_pipeline_xx_5.5.1_3.0_1738785212416.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_multilingual_uncased_finetuned_wikipedia_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_multilingual_uncased_finetuned_wikipedia_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_multilingual_uncased_finetuned_wikipedia_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|625.6 MB|

## References

https://huggingface.co/rman-rahimi-29/bert-base-multilingual-uncased-finetuned-Wikipedia

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings