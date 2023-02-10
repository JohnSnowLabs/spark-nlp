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
annotator: Pipeline
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Typed Dependency parser, trained on the on the CONLL dataset. 

Dependency parsing is the task of extracting a dependency parse of a sentence that represents its grammatical structure and defines the relationships between “head” words and words, which modify those heads.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_parse_en_4.0.0_3.0_1656456276940.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dependency_parse_en_4.0.0_3.0_1656456276940.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

## Results

```bash
+---------------------------------------------------------------------------------+--------------------------------------------------------+
|result                                                                           |result                                                  |
+---------------------------------------------------------------------------------+--------------------------------------------------------+
|[ROOT, Dependencies, represents, words, relationships, Sentence, Sentence, words]|[root, parataxis, nsubj, amod, nsubj, case, nsubj, flat]|
+---------------------------------------------------------------------------------+--------------------------------------------------------+

```

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
