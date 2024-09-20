---
layout: model
title: Persian tara_roberta_base_persian_farsi_qa_pipeline pipeline RoBertaForQuestionAnswering from hosseinhimself
author: John Snow Labs
name: tara_roberta_base_persian_farsi_qa_pipeline
date: 2024-09-04
tags: [fa, open_source, pipeline, onnx]
task: Question Answering
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tara_roberta_base_persian_farsi_qa_pipeline` is a Persian model originally trained by hosseinhimself.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tara_roberta_base_persian_farsi_qa_pipeline_fa_5.5.0_3.0_1725450771187.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tara_roberta_base_persian_farsi_qa_pipeline_fa_5.5.0_3.0_1725450771187.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tara_roberta_base_persian_farsi_qa_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tara_roberta_base_persian_farsi_qa_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tara_roberta_base_persian_farsi_qa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|463.6 MB|

## References

https://huggingface.co/hosseinhimself/tara-roberta-base-fa-qa

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering