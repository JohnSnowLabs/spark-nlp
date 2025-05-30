---
layout: model
title: English xlm_roberta_base_finetuned_germeval_14_4_labels_pipeline pipeline XlmRoBertaForTokenClassification from stefanieZ
author: John Snow Labs
name: xlm_roberta_base_finetuned_germeval_14_4_labels_pipeline
date: 2024-06-11
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_base_finetuned_germeval_14_4_labels_pipeline` is a English model originally trained by stefanieZ.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_finetuned_germeval_14_4_labels_pipeline_en_5.4.0_3.0_1718098725921.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_finetuned_germeval_14_4_labels_pipeline_en_5.4.0_3.0_1718098725921.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_base_finetuned_germeval_14_4_labels_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_base_finetuned_germeval_14_4_labels_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_finetuned_germeval_14_4_labels_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|861.2 MB|

## References

https://huggingface.co/stefanieZ/xlm-roberta-base-finetuned-germeval-14-4-labels

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification