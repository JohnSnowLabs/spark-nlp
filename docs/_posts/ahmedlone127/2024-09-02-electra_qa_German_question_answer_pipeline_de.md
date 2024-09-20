---
layout: model
title: German electra_qa_German_question_answer_pipeline pipeline BertForQuestionAnswering from Sahajtomar
author: John Snow Labs
name: electra_qa_German_question_answer_pipeline
date: 2024-09-02
tags: [de, open_source, pipeline, onnx]
task: Question Answering
language: de
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`electra_qa_German_question_answer_pipeline` is a German model originally trained by Sahajtomar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_qa_German_question_answer_pipeline_de_5.5.0_3.0_1725312632360.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/electra_qa_German_question_answer_pipeline_de_5.5.0_3.0_1725312632360.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("electra_qa_German_question_answer_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("electra_qa_German_question_answer_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_qa_German_question_answer_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|1.3 GB|

## References

https://huggingface.co/Sahajtomar/German-question-answer-Electra

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering