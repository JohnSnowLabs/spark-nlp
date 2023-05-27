---
layout: model
title: Typed Dependency Parsing pipeline for English
author: John Snow Labs
name: dependency_parse
date: 2023-05-27
tags: [pipeline, dependency_parsing, untyped_dependency_parsing, typed_dependency_parsing, laballed_depdency_parsing, unlaballed_depdency_parsing, en, open_source]
task: Dependency Parser
language: en
edition: Spark NLP 4.4.2
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Typed Dependency parser, trained on the on the CONLL dataset. 

Dependency parsing is the task of extracting a dependency parse of a sentence that represents its grammatical structure and defines the relationships between “head” words and words, which modify those heads.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_parse_en_4.4.2_3.2_1685184038289.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dependency_parse_en_4.4.2_3.2_1685184038289.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline('dependency_parse', lang = 'en')
annotations =  pipeline.fullAnnotate("Dependencies represents relationships betweens words in a Sentence "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("dependency_parse", lang = "en")
val result = pipeline.fullAnnotate("Dependencies represents relationships betweens words in a Sentence")(0)

```

{:.nlu-block}
```python

nlu.load("dep.typed").predict("Dependencies represents relationships betweens words in a Sentence")

    
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline('dependency_parse', lang = 'en')
annotations =  pipeline.fullAnnotate("Dependencies represents relationships betweens words in a Sentence "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("dependency_parse", lang = "en")
val result = pipeline.fullAnnotate("Dependencies represents relationships betweens words in a Sentence")(0)
```

{:.nlu-block}
```python
nlu.load("dep.typed").predict("Dependencies represents relationships betweens words in a Sentence")
```
</div>

## Results

```bash
Results


+---------------------------------------------------------------------------------+--------------------------------------------------------+
|result                                                                           |result                                                  |
+---------------------------------------------------------------------------------+--------------------------------------------------------+
|[ROOT, Dependencies, represents, words, relationships, Sentence, Sentence, words]|[root, parataxis, nsubj, amod, nsubj, case, nsubj, flat]|
+---------------------------------------------------------------------------------+--------------------------------------------------------+



{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dependency_parse|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|23.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- PerceptronModel
- DependencyParserModel
- TypedDependencyParserModel