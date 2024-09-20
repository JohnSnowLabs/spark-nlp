---
layout: model
title: English unifiedqa_v2_t5_base_1363200_finetuned_qa_doqa_pipeline pipeline T5Transformer from makanju0la
author: John Snow Labs
name: unifiedqa_v2_t5_base_1363200_finetuned_qa_doqa_pipeline
date: 2024-08-21
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`unifiedqa_v2_t5_base_1363200_finetuned_qa_doqa_pipeline` is a English model originally trained by makanju0la.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/unifiedqa_v2_t5_base_1363200_finetuned_qa_doqa_pipeline_en_5.4.2_3.0_1724242130462.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/unifiedqa_v2_t5_base_1363200_finetuned_qa_doqa_pipeline_en_5.4.2_3.0_1724242130462.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("unifiedqa_v2_t5_base_1363200_finetuned_qa_doqa_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("unifiedqa_v2_t5_base_1363200_finetuned_qa_doqa_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|unifiedqa_v2_t5_base_1363200_finetuned_qa_doqa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|961.5 MB|

## References

https://huggingface.co/makanju0la/unifiedqa-v2-t5-base-1363200-finetuned-qa-doqa

## Included Models

- DocumentAssembler
- T5Transformer