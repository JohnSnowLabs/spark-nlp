---
layout: model
title: Malay (macrolanguage) t5_finetune_paraphrase_tiny_standard_bahasa_cased_pipeline pipeline T5Transformer from mesolitica
author: John Snow Labs
name: t5_finetune_paraphrase_tiny_standard_bahasa_cased_pipeline
date: 2024-07-29
tags: [ms, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ms
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`t5_finetune_paraphrase_tiny_standard_bahasa_cased_pipeline` is a Malay (macrolanguage) model originally trained by mesolitica.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_finetune_paraphrase_tiny_standard_bahasa_cased_pipeline_ms_5.4.2_3.0_1722266794454.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_finetune_paraphrase_tiny_standard_bahasa_cased_pipeline_ms_5.4.2_3.0_1722266794454.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("t5_finetune_paraphrase_tiny_standard_bahasa_cased_pipeline", lang = "ms")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("t5_finetune_paraphrase_tiny_standard_bahasa_cased_pipeline", lang = "ms")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_finetune_paraphrase_tiny_standard_bahasa_cased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ms|
|Size:|222.8 MB|

## References

https://huggingface.co/mesolitica/finetune-paraphrase-t5-tiny-standard-bahasa-cased

## Included Models

- DocumentAssembler
- T5Transformer