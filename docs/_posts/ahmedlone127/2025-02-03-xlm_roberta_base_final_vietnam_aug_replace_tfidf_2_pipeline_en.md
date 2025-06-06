---
layout: model
title: English xlm_roberta_base_final_vietnam_aug_replace_tfidf_2_pipeline pipeline XlmRoBertaForSequenceClassification from ThuyNT03
author: John Snow Labs
name: xlm_roberta_base_final_vietnam_aug_replace_tfidf_2_pipeline
date: 2025-02-03
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_base_final_vietnam_aug_replace_tfidf_2_pipeline` is a English model originally trained by ThuyNT03.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_final_vietnam_aug_replace_tfidf_2_pipeline_en_5.5.1_3.0_1738619609937.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_final_vietnam_aug_replace_tfidf_2_pipeline_en_5.5.1_3.0_1738619609937.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_base_final_vietnam_aug_replace_tfidf_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_base_final_vietnam_aug_replace_tfidf_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_final_vietnam_aug_replace_tfidf_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|793.7 MB|

## References

https://huggingface.co/ThuyNT03/xlm-roberta-base-Final_VietNam-aug_replace_tfidf-2

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification