---
layout: model
title: Danish sent_meda_bert_den_intelligente_patientjournal_pipeline pipeline BertSentenceEmbeddings from Den-Intelligente-Patientjournal
author: John Snow Labs
name: sent_meda_bert_den_intelligente_patientjournal_pipeline
date: 2025-01-29
tags: [da, open_source, pipeline, onnx]
task: Embeddings
language: da
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_meda_bert_den_intelligente_patientjournal_pipeline` is a Danish model originally trained by Den-Intelligente-Patientjournal.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_meda_bert_den_intelligente_patientjournal_pipeline_da_5.5.1_3.0_1738153043473.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_meda_bert_den_intelligente_patientjournal_pipeline_da_5.5.1_3.0_1738153043473.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_meda_bert_den_intelligente_patientjournal_pipeline", lang = "da")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_meda_bert_den_intelligente_patientjournal_pipeline", lang = "da")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_meda_bert_den_intelligente_patientjournal_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|da|
|Size:|412.8 MB|

## References

https://huggingface.co/Den-Intelligente-Patientjournal/MeDa-BERT

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings