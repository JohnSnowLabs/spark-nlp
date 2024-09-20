---
layout: model
title: English xlm_robert_finetune_model_thai_french_mm_pipeline pipeline XlmRoBertaForTokenClassification from zhangwenzhe
author: John Snow Labs
name: xlm_robert_finetune_model_thai_french_mm_pipeline
date: 2024-09-20
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_robert_finetune_model_thai_french_mm_pipeline` is a English model originally trained by zhangwenzhe.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_robert_finetune_model_thai_french_mm_pipeline_en_5.5.0_3.0_1726844575810.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_robert_finetune_model_thai_french_mm_pipeline_en_5.5.0_3.0_1726844575810.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_robert_finetune_model_thai_french_mm_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_robert_finetune_model_thai_french_mm_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_robert_finetune_model_thai_french_mm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|773.2 MB|

## References

https://huggingface.co/zhangwenzhe/XLM-Robert-finetune-model-THAI-FR-MM

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification