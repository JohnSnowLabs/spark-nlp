---
layout: model
title: Dutch, Flemish sent_bert_base_dutch_cased_finetuned_germanic_languages_pipeline pipeline BertSentenceEmbeddings from GeniusVoice
author: John Snow Labs
name: sent_bert_base_dutch_cased_finetuned_germanic_languages_pipeline
date: 2025-02-03
tags: [nl, open_source, pipeline, onnx]
task: Embeddings
language: nl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_base_dutch_cased_finetuned_germanic_languages_pipeline` is a Dutch, Flemish model originally trained by GeniusVoice.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_dutch_cased_finetuned_germanic_languages_pipeline_nl_5.5.1_3.0_1738572010900.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_dutch_cased_finetuned_germanic_languages_pipeline_nl_5.5.1_3.0_1738572010900.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_base_dutch_cased_finetuned_germanic_languages_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_base_dutch_cased_finetuned_germanic_languages_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_dutch_cased_finetuned_germanic_languages_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|407.3 MB|

## References

https://huggingface.co/GeniusVoice/bert-base-dutch-cased-finetuned-gem

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings