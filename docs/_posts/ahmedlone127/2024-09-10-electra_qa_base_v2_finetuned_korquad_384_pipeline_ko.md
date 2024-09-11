---
layout: model
title: Korean electra_qa_base_v2_finetuned_korquad_384_pipeline pipeline BertForQuestionAnswering from monologg
author: John Snow Labs
name: electra_qa_base_v2_finetuned_korquad_384_pipeline
date: 2024-09-10
tags: [ko, open_source, pipeline, onnx]
task: Question Answering
language: ko
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`electra_qa_base_v2_finetuned_korquad_384_pipeline` is a Korean model originally trained by monologg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_qa_base_v2_finetuned_korquad_384_pipeline_ko_5.5.0_3.0_1725926778058.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/electra_qa_base_v2_finetuned_korquad_384_pipeline_ko_5.5.0_3.0_1725926778058.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("electra_qa_base_v2_finetuned_korquad_384_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("electra_qa_base_v2_finetuned_korquad_384_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_qa_base_v2_finetuned_korquad_384_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|411.8 MB|

## References

https://huggingface.co/monologg/koelectra-base-v2-finetuned-korquad-384

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering