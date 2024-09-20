---
layout: model
title: English xlmroberta_finetuned_recipeqa_modified_pipeline pipeline XlmRoBertaForQuestionAnswering from tamhuynh27
author: John Snow Labs
name: xlmroberta_finetuned_recipeqa_modified_pipeline
date: 2024-09-06
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

Pretrained XlmRoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_finetuned_recipeqa_modified_pipeline` is a English model originally trained by tamhuynh27.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_finetuned_recipeqa_modified_pipeline_en_5.5.0_3.0_1725640874796.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_finetuned_recipeqa_modified_pipeline_en_5.5.0_3.0_1725640874796.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_finetuned_recipeqa_modified_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_finetuned_recipeqa_modified_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_finetuned_recipeqa_modified_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|856.8 MB|

## References

https://huggingface.co/tamhuynh27/xlmroberta-finetuned-recipeqa-modified

## Included Models

- MultiDocumentAssembler
- XlmRoBertaForQuestionAnswering