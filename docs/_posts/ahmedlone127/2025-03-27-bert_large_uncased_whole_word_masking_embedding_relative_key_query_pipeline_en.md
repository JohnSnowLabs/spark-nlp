---
layout: model
title: English bert_large_uncased_whole_word_masking_embedding_relative_key_query_pipeline pipeline BertEmbeddings from zhiheng-huang
author: John Snow Labs
name: bert_large_uncased_whole_word_masking_embedding_relative_key_query_pipeline
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_large_uncased_whole_word_masking_embedding_relative_key_query_pipeline` is a English model originally trained by zhiheng-huang.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_uncased_whole_word_masking_embedding_relative_key_query_pipeline_en_5.5.1_3.0_1743081034532.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_large_uncased_whole_word_masking_embedding_relative_key_query_pipeline_en_5.5.1_3.0_1743081034532.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_large_uncased_whole_word_masking_embedding_relative_key_query_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_large_uncased_whole_word_masking_embedding_relative_key_query_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_large_uncased_whole_word_masking_embedding_relative_key_query_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/zhiheng-huang/bert-large-uncased-whole-word-masking-embedding-relative-key-query

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings