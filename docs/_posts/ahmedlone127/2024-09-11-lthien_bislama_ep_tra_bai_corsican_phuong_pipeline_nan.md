---
layout: model
title: None lthien_bislama_ep_tra_bai_corsican_phuong_pipeline pipeline DistilBertForQuestionAnswering from hi113
author: John Snow Labs
name: lthien_bislama_ep_tra_bai_corsican_phuong_pipeline
date: 2024-09-11
tags: [nan, open_source, pipeline, onnx]
task: Question Answering
language: nan
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`lthien_bislama_ep_tra_bai_corsican_phuong_pipeline` is a None model originally trained by hi113.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lthien_bislama_ep_tra_bai_corsican_phuong_pipeline_nan_5.5.0_3.0_1726088239959.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/lthien_bislama_ep_tra_bai_corsican_phuong_pipeline_nan_5.5.0_3.0_1726088239959.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("lthien_bislama_ep_tra_bai_corsican_phuong_pipeline", lang = "nan")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("lthien_bislama_ep_tra_bai_corsican_phuong_pipeline", lang = "nan")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|lthien_bislama_ep_tra_bai_corsican_phuong_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nan|
|Size:|247.2 MB|

## References

https://huggingface.co/hi113/ltHien_Bi_Ep_Tra_Bai_Co_Phuong

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering