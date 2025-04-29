---
layout: model
title: Multilingual sent_xtreme_squad_bert_base_multilingual_cased_pipeline pipeline BertSentenceEmbeddings from dyyyyyyyy
author: John Snow Labs
name: sent_xtreme_squad_bert_base_multilingual_cased_pipeline
date: 2025-04-07
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

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_xtreme_squad_bert_base_multilingual_cased_pipeline` is a Multilingual model originally trained by dyyyyyyyy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_xtreme_squad_bert_base_multilingual_cased_pipeline_xx_5.5.1_3.0_1744007645452.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_xtreme_squad_bert_base_multilingual_cased_pipeline_xx_5.5.1_3.0_1744007645452.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_xtreme_squad_bert_base_multilingual_cased_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_xtreme_squad_bert_base_multilingual_cased_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_xtreme_squad_bert_base_multilingual_cased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|665.6 MB|

## References

https://huggingface.co/dyyyyyyyy/XTREME_squad_BERT-base-multilingual-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings