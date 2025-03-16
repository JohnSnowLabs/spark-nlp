---
layout: model
title: English sent_bert_uncased_l_2_h_256_a_4_mlm_multi_emails_hq_pipeline pipeline BertSentenceEmbeddings from postbot
author: John Snow Labs
name: sent_bert_uncased_l_2_h_256_a_4_mlm_multi_emails_hq_pipeline
date: 2025-01-26
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

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_uncased_l_2_h_256_a_4_mlm_multi_emails_hq_pipeline` is a English model originally trained by postbot.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_uncased_l_2_h_256_a_4_mlm_multi_emails_hq_pipeline_en_5.5.1_3.0_1737855950368.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_uncased_l_2_h_256_a_4_mlm_multi_emails_hq_pipeline_en_5.5.1_3.0_1737855950368.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_uncased_l_2_h_256_a_4_mlm_multi_emails_hq_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_uncased_l_2_h_256_a_4_mlm_multi_emails_hq_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_uncased_l_2_h_256_a_4_mlm_multi_emails_hq_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|36.5 MB|

## References

https://huggingface.co/postbot/bert_uncased_L-2_H-256_A-4-mlm-multi-emails-hq

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings