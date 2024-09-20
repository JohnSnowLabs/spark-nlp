---
layout: model
title: English xlmroberta_finetuned_tydiqa_pipeline pipeline XlmRoBertaForQuestionAnswering from Auracle7
author: John Snow Labs
name: xlmroberta_finetuned_tydiqa_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained XlmRoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_finetuned_tydiqa_pipeline` is a English model originally trained by Auracle7.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_finetuned_tydiqa_pipeline_en_5.4.2_3.0_1725173229851.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_finetuned_tydiqa_pipeline_en_5.4.2_3.0_1725173229851.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_finetuned_tydiqa_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_finetuned_tydiqa_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_finetuned_tydiqa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|848.9 MB|

## References

https://huggingface.co/Auracle7/XLMRoberta-finetuned-TyDIQA

## Included Models

- MultiDocumentAssembler
- XlmRoBertaForQuestionAnswering