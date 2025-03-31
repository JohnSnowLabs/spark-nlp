---
layout: model
title: English finetuned_bert_mrpc_snape_v_pipeline pipeline BertForSequenceClassification from Snape-v
author: John Snow Labs
name: finetuned_bert_mrpc_snape_v_pipeline
date: 2025-03-31
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_bert_mrpc_snape_v_pipeline` is a English model originally trained by Snape-v.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_bert_mrpc_snape_v_pipeline_en_5.5.1_3.0_1743442138223.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_bert_mrpc_snape_v_pipeline_en_5.5.1_3.0_1743442138223.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_bert_mrpc_snape_v_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_bert_mrpc_snape_v_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_bert_mrpc_snape_v_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|405.9 MB|

## References

https://huggingface.co/Snape-v/finetuned-bert-mrpc

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification