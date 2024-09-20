---
layout: model
title: English squad_trained_ep4_batch16_finetuned_squad_emrqa_msquad_pipeline pipeline DistilBertForQuestionAnswering from wieheistdu
author: John Snow Labs
name: squad_trained_ep4_batch16_finetuned_squad_emrqa_msquad_pipeline
date: 2024-09-11
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`squad_trained_ep4_batch16_finetuned_squad_emrqa_msquad_pipeline` is a English model originally trained by wieheistdu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/squad_trained_ep4_batch16_finetuned_squad_emrqa_msquad_pipeline_en_5.5.0_3.0_1726017054189.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/squad_trained_ep4_batch16_finetuned_squad_emrqa_msquad_pipeline_en_5.5.0_3.0_1726017054189.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("squad_trained_ep4_batch16_finetuned_squad_emrqa_msquad_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("squad_trained_ep4_batch16_finetuned_squad_emrqa_msquad_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|squad_trained_ep4_batch16_finetuned_squad_emrqa_msquad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/wieheistdu/squad-trained-ep4-batch16-finetuned-squad-emrQA-msquad

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering