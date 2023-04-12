---
layout: model
title: Explain Document pipeline for Korean (explain_document_lg)
author: John Snow Labs
name: explain_document_lg
date: 2021-04-30
tags: [korean, open_source, explain_document_lg, pipeline, ko, ner]
task: Named Entity Recognition
language: ko
edition: Spark NLP 3.0.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_lg is a pre-trained pipeline that we can use to process text with a simple pipeline that performs basic processing steps and recognizes entities. It performs most of the common text processing tasks on your dataframe

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_ko_3.0.2_3.0_1619772353571.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_lg_ko_3.0.2_3.0_1619772353571.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline('explain_document_lg', lang = 'ko')
annotations =  pipeline.fullAnnotate(""안녕하세요, 환영합니다!"")[0]
annotations.keys()

```
```scala
val pipeline = new PretrainedPipeline("explain_document_lg", lang = "ko")
val result = pipeline.fullAnnotate("안녕하세요, 환영합니다!")(0)
```


{:.nlu-block}
```python
import nlu
nlu.load("ko.explain_document").predict("""안녕하세요, 환영합니다!""")
```

</div>

## Results

```bash
+------------------------+--------------------------+--------------------------+--------------------------------+----------------------------+---------------------+
|text                      |document            |sentence              |token                           |ner                           |ner_chunk      |
+------------------------+--------------------------+--------------------------+--------------------------------+----------------------------+---------------------+
|안녕, 존 스노우!|[안녕, 존 스노우!]|[안녕, 존 스노우!]|[안녕, ,, 존, 스노우, !]   |[B-DATE, O, O, O, O]| [안녕]            |
+------------------------+--------------------------+--------------------------+--------------------------------+----------------------------+---------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_lg|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|

## Included Models

- DocumentAssembler
- SentenceDetector
- WordSegmenterModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter