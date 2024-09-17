---
layout: model
title: English simpletransformer_qa_roberta_base_mtanzi_hopin_pipeline pipeline RoBertaForQuestionAnswering from mtanzi-hopin
author: John Snow Labs
name: simpletransformer_qa_roberta_base_mtanzi_hopin_pipeline
date: 2024-09-15
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`simpletransformer_qa_roberta_base_mtanzi_hopin_pipeline` is a English model originally trained by mtanzi-hopin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/simpletransformer_qa_roberta_base_mtanzi_hopin_pipeline_en_5.5.0_3.0_1726379655532.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/simpletransformer_qa_roberta_base_mtanzi_hopin_pipeline_en_5.5.0_3.0_1726379655532.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("simpletransformer_qa_roberta_base_mtanzi_hopin_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("simpletransformer_qa_roberta_base_mtanzi_hopin_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|simpletransformer_qa_roberta_base_mtanzi_hopin_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|414.2 MB|

## References

https://huggingface.co/mtanzi-hopin/simpletransformer-qa-roberta-base

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering