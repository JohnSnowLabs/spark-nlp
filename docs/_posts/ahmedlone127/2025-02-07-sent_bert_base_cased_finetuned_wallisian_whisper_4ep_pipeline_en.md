---
layout: model
title: English sent_bert_base_cased_finetuned_wallisian_whisper_4ep_pipeline pipeline BertSentenceEmbeddings from btamm12
author: John Snow Labs
name: sent_bert_base_cased_finetuned_wallisian_whisper_4ep_pipeline
date: 2025-02-07
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

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_base_cased_finetuned_wallisian_whisper_4ep_pipeline` is a English model originally trained by btamm12.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_cased_finetuned_wallisian_whisper_4ep_pipeline_en_5.5.1_3.0_1738947489225.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_cased_finetuned_wallisian_whisper_4ep_pipeline_en_5.5.1_3.0_1738947489225.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_base_cased_finetuned_wallisian_whisper_4ep_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_base_cased_finetuned_wallisian_whisper_4ep_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_cased_finetuned_wallisian_whisper_4ep_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|404.2 MB|

## References

https://huggingface.co/btamm12/bert-base-cased-finetuned-wls-whisper-4ep

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings