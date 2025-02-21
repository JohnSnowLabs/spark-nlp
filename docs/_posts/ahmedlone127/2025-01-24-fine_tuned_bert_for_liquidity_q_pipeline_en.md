---
layout: model
title: English fine_tuned_bert_for_liquidity_q_pipeline pipeline BertEmbeddings from punchnami
author: John Snow Labs
name: fine_tuned_bert_for_liquidity_q_pipeline
date: 2025-01-24
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fine_tuned_bert_for_liquidity_q_pipeline` is a English model originally trained by punchnami.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fine_tuned_bert_for_liquidity_q_pipeline_en_5.5.1_3.0_1737708078852.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fine_tuned_bert_for_liquidity_q_pipeline_en_5.5.1_3.0_1737708078852.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fine_tuned_bert_for_liquidity_q_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fine_tuned_bert_for_liquidity_q_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fine_tuned_bert_for_liquidity_q_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.2 MB|

## References

https://huggingface.co/punchnami/fine-tuned-BERT-for-liquidity-q

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings