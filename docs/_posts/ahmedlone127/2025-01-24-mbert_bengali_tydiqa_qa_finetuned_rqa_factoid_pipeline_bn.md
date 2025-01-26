---
layout: model
title: Bengali mbert_bengali_tydiqa_qa_finetuned_rqa_factoid_pipeline pipeline BertForQuestionAnswering from AsifAbrar6
author: John Snow Labs
name: mbert_bengali_tydiqa_qa_finetuned_rqa_factoid_pipeline
date: 2025-01-24
tags: [bn, open_source, pipeline, onnx]
task: Question Answering
language: bn
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mbert_bengali_tydiqa_qa_finetuned_rqa_factoid_pipeline` is a Bengali model originally trained by AsifAbrar6.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mbert_bengali_tydiqa_qa_finetuned_rqa_factoid_pipeline_bn_5.5.1_3.0_1737757942106.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mbert_bengali_tydiqa_qa_finetuned_rqa_factoid_pipeline_bn_5.5.1_3.0_1737757942106.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mbert_bengali_tydiqa_qa_finetuned_rqa_factoid_pipeline", lang = "bn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mbert_bengali_tydiqa_qa_finetuned_rqa_factoid_pipeline", lang = "bn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mbert_bengali_tydiqa_qa_finetuned_rqa_factoid_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|bn|
|Size:|625.5 MB|

## References

https://huggingface.co/AsifAbrar6/mbert-bengali-tydiqa-qa-finetuned-RQA-factoid

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering