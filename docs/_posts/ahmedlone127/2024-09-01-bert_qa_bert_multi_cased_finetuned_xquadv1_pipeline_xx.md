---
layout: model
title: Multilingual bert_qa_bert_multi_cased_finetuned_xquadv1_pipeline pipeline BertForQuestionAnswering from mrm8488
author: John Snow Labs
name: bert_qa_bert_multi_cased_finetuned_xquadv1_pipeline
date: 2024-09-01
tags: [xx, open_source, pipeline, onnx]
task: Question Answering
language: xx
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_qa_bert_multi_cased_finetuned_xquadv1_pipeline` is a Multilingual model originally trained by mrm8488.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_bert_multi_cased_finetuned_xquadv1_pipeline_xx_5.4.2_3.0_1725150995500.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_qa_bert_multi_cased_finetuned_xquadv1_pipeline_xx_5.4.2_3.0_1725150995500.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_qa_bert_multi_cased_finetuned_xquadv1_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_qa_bert_multi_cased_finetuned_xquadv1_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_bert_multi_cased_finetuned_xquadv1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|665.1 MB|

## References

https://huggingface.co/mrm8488/bert-multi-cased-finetuned-xquadv1

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering