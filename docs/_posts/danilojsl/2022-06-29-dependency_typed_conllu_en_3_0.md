---
layout: model
title: Typed Dependency Parsing for English
author: John Snow Labs
name: dependency_typed_conllu
date: 2022-06-29
tags: [typed_dependency_parsing, labelled_dependency_parsing, dependency_parsing, en, open_source]
task: Dependency Parser
language: en
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Typed Dependency parser, trained on the on the CONLL dataset.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_typed_conllu_en_3.4.4_3.0_1656517004297.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.annotators import *

documentAssembler     = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector      = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer             = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
posTagger             = PerceptronModel.pretrained().setInputCols(["token", "sentence"]).setOutputCol("pos")
dependencyParser      = DependencyParserModel.pretrained().setInputCols(["sentence", "pos", "token"]).setOutputCol("dependency")
typedDependencyParser = TypedDependencyParserModel.pretrained().setInputCols(["token", "pos", "dependency"]).setOutputCol("labdep")
pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, posTagger, dependencyParser, typedDependencyParser])
data = spark.createDataFrame({"text": "Dependencies represents relationships betweens words in a Sentence"})
# Create data frame
df = spark.createDataFrame(data)
result = pipeline.fit(df).transform(df)
result.select("dependency.result", "labdep.result").show(false)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dependency_typed_conllu|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, pos, dep_root]|
|Output Labels:|[dep_mod]|
|Language:|en|
|Size:|2.5 MB|