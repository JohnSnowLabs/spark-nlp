---
layout: model
title: Portuguese sent_bert_base_portuguese_cased_neuralmind_pipeline pipeline BertSentenceEmbeddings from neuralmind
author: John Snow Labs
name: sent_bert_base_portuguese_cased_neuralmind_pipeline
date: 2024-08-31
tags: [pt, open_source, pipeline, onnx]
task: Embeddings
language: pt
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_base_portuguese_cased_neuralmind_pipeline` is a Portuguese model originally trained by neuralmind.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_portuguese_cased_neuralmind_pipeline_pt_5.4.2_3.0_1725121286260.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_portuguese_cased_neuralmind_pipeline_pt_5.4.2_3.0_1725121286260.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_base_portuguese_cased_neuralmind_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_base_portuguese_cased_neuralmind_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_portuguese_cased_neuralmind_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|406.5 MB|

## References

https://huggingface.co/neuralmind/bert-base-portuguese-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings