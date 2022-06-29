---
layout: model
title: Untyped Dependency Parsing for English
author: John Snow Labs
name: dependency_conllu
date: 2022-06-29
tags: [dependency_parsing, unlabelled_dependency_parsing, untyped_dependency_parsing, en, open_source]
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

Untyped Dependency parser, trained on the on the CONLL dataset.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_conllu_en_3.4.4_3.0_1656516113100.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
result.select("dependency.result").show(false)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dependency_conllu|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, pos, token]|
|Output Labels:|[dep_root]|
|Language:|en|
|Size:|17.5 MB|