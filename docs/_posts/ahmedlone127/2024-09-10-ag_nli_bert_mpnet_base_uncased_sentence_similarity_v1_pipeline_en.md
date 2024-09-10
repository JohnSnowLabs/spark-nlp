---
layout: model
title: English ag_nli_bert_mpnet_base_uncased_sentence_similarity_v1_pipeline pipeline MPNetEmbeddings from abbasgolestani
author: John Snow Labs
name: ag_nli_bert_mpnet_base_uncased_sentence_similarity_v1_pipeline
date: 2024-09-10
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

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ag_nli_bert_mpnet_base_uncased_sentence_similarity_v1_pipeline` is a English model originally trained by abbasgolestani.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ag_nli_bert_mpnet_base_uncased_sentence_similarity_v1_pipeline_en_5.5.0_3.0_1725963304483.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ag_nli_bert_mpnet_base_uncased_sentence_similarity_v1_pipeline_en_5.5.0_3.0_1725963304483.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ag_nli_bert_mpnet_base_uncased_sentence_similarity_v1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ag_nli_bert_mpnet_base_uncased_sentence_similarity_v1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ag_nli_bert_mpnet_base_uncased_sentence_similarity_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|405.5 MB|

## References

https://huggingface.co/abbasgolestani/ag-nli-bert-mpnet-base-uncased-sentence-similarity-v1

## Included Models

- DocumentAssembler
- MPNetEmbeddings