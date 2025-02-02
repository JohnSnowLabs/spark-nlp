---
layout: model
title: English sent_msmarco_v2_bert_mlm_csv300k_pipeline pipeline BertSentenceEmbeddings from pxyu
author: John Snow Labs
name: sent_msmarco_v2_bert_mlm_csv300k_pipeline
date: 2025-02-02
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

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_msmarco_v2_bert_mlm_csv300k_pipeline` is a English model originally trained by pxyu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_msmarco_v2_bert_mlm_csv300k_pipeline_en_5.5.1_3.0_1738466776892.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_msmarco_v2_bert_mlm_csv300k_pipeline_en_5.5.1_3.0_1738466776892.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_msmarco_v2_bert_mlm_csv300k_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_msmarco_v2_bert_mlm_csv300k_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_msmarco_v2_bert_mlm_csv300k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|5.6 MB|

## References

https://huggingface.co/pxyu/MSMARCO-V2-BERT-MLM-CSV300k

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings