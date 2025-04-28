---
layout: model
title: Tamil xlm_roberta_base_fintuned_panx_tamil_hindi_pipeline pipeline XlmRoBertaForTokenClassification from Lokeshwaran
author: John Snow Labs
name: xlm_roberta_base_fintuned_panx_tamil_hindi_pipeline
date: 2025-01-30
tags: [ta, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ta
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_base_fintuned_panx_tamil_hindi_pipeline` is a Tamil model originally trained by Lokeshwaran.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_fintuned_panx_tamil_hindi_pipeline_ta_5.5.1_3.0_1738252941890.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_fintuned_panx_tamil_hindi_pipeline_ta_5.5.1_3.0_1738252941890.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_base_fintuned_panx_tamil_hindi_pipeline", lang = "ta")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_base_fintuned_panx_tamil_hindi_pipeline", lang = "ta")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_fintuned_panx_tamil_hindi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ta|
|Size:|828.1 MB|

## References

https://huggingface.co/Lokeshwaran/xlm-roberta-base-fintuned-panx-ta-hi

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification