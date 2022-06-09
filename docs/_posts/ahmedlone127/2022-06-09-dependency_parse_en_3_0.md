---
layout: model
title: Typed Dependency Parsing pipeline for English
author: ahmedlone127
name: dependency_parse
date: 2022-06-09
tags: [pipeline, dependency_parsing, untyped_dependency_parsing, typed_dependency_parsing, laballed_depdency_parsing, unlaballed_depdency_parsing, en, open_source]
task: Dependency Parser
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Typed Dependency parser, trained on the on the CONLL dataset. 

Dependency parsing is the task of extracting a dependency parse of a sentence that represents its grammatical structure and defines the relationships between “head” words and words, which modify those heads.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/ahmedlone127/dependency_parse_en_4.0.0_3.0_1654792687447.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


from sparknlp.pretrained import PretrainedPipelinein
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
|Edition:|Community|
|Language:|en|
|Size:|29.4 KB|

## Included Models

- DocumentAssembler
- TokenizerModel
- NormalizerModel