---
layout: model
title: English sent_bert_large_cased_sigir_support_refute_norwegian_label_40_2nd_test_lr10_8_fast_8_pipeline pipeline BertSentenceEmbeddings from jojoUla
author: John Snow Labs
name: sent_bert_large_cased_sigir_support_refute_norwegian_label_40_2nd_test_lr10_8_fast_8_pipeline
date: 2024-09-23
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

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_large_cased_sigir_support_refute_norwegian_label_40_2nd_test_lr10_8_fast_8_pipeline` is a English model originally trained by jojoUla.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_large_cased_sigir_support_refute_norwegian_label_40_2nd_test_lr10_8_fast_8_pipeline_en_5.5.0_3.0_1727113815073.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_large_cased_sigir_support_refute_norwegian_label_40_2nd_test_lr10_8_fast_8_pipeline_en_5.5.0_3.0_1727113815073.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_large_cased_sigir_support_refute_norwegian_label_40_2nd_test_lr10_8_fast_8_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_large_cased_sigir_support_refute_norwegian_label_40_2nd_test_lr10_8_fast_8_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_large_cased_sigir_support_refute_norwegian_label_40_2nd_test_lr10_8_fast_8_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/jojoUla/bert-large-cased-sigir-support-refute-no-label-40-2nd-test-LR10-8-fast-8

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings