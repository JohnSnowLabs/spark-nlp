---
layout: model
title: English q_anda_bert_finetuned_squad_accelerate_pipeline pipeline BertForQuestionAnswering from kaiku03
author: John Snow Labs
name: q_anda_bert_finetuned_squad_accelerate_pipeline
date: 2025-01-30
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`q_anda_bert_finetuned_squad_accelerate_pipeline` is a English model originally trained by kaiku03.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/q_anda_bert_finetuned_squad_accelerate_pipeline_en_5.5.1_3.0_1738272630957.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/q_anda_bert_finetuned_squad_accelerate_pipeline_en_5.5.1_3.0_1738272630957.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("q_anda_bert_finetuned_squad_accelerate_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("q_anda_bert_finetuned_squad_accelerate_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|q_anda_bert_finetuned_squad_accelerate_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|403.7 MB|

## References

https://huggingface.co/kaiku03/Q-andA-bert-finetuned-squad-accelerate

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering