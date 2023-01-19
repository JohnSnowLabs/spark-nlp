---
layout: model
title: Match Pattern
author: John Snow Labs
name: match_pattern
date: 2022-06-19
tags: [en, open_source]
task: Text Classification
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The match_pattern is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps and matches pattrens  .
	It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/match_pattern_en_4.0.0_3.0_1655653735848.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/match_pattern_en_4.0.0_3.0_1655653735848.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("match_pattern", "en", "clinical/models")
	result = pipeline.annotate("""I love johnsnowlabs!  """)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|match_pattern|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|29.0 KB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- RegexMatcherModel