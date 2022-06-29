---
layout: model
title: Typed Dependency Parsing pipeline for English
author: John Snow Labs
name: dependency_parse
date: 2022-06-29
tags: [pipeline, dependency_parsing, untyped_dependency_parsing, typed_dependency_parsing, laballed_depdency_parsing, unlaballed_depdency_parsing, en, open_source]
task: Dependency Parser
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Typed Dependency parser, trained on the on the CONLL dataset.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_parse_en_4.0.0_3.0_1656518536824.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('dependency_parse', lang = 'en')
annotations =  pipeline.fullAnnotate("Dependencies represents relationships betweens words in a Sentence "")[0]
annotations.keys()

```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dependency_parse|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|24.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- PerceptronModel
- DependencyParserModel
- TypedDependencyParserModel