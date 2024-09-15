---
layout: model
title: Multilingual distilbert_base_multilingual_cased_qa_squad_v1_norwegian_bokml_pipeline pipeline DistilBertForQuestionAnswering from eanderson
author: John Snow Labs
name: distilbert_base_multilingual_cased_qa_squad_v1_norwegian_bokml_pipeline
date: 2024-09-11
tags: [xx, open_source, pipeline, onnx]
task: Question Answering
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_base_multilingual_cased_qa_squad_v1_norwegian_bokml_pipeline` is a Multilingual model originally trained by eanderson.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_qa_squad_v1_norwegian_bokml_pipeline_xx_5.5.0_3.0_1726028609992.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_base_multilingual_cased_qa_squad_v1_norwegian_bokml_pipeline_xx_5.5.0_3.0_1726028609992.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_base_multilingual_cased_qa_squad_v1_norwegian_bokml_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_base_multilingual_cased_qa_squad_v1_norwegian_bokml_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_multilingual_cased_qa_squad_v1_norwegian_bokml_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|505.4 MB|

## References

https://huggingface.co/eanderson/distilbert-base-multilingual-cased-qa-squad_v1_nb

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering