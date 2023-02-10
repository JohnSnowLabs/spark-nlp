---
layout: model
title: Untyped Dependency Parsing for English
author: John Snow Labs
name: dependency_conllu
date: 2021-03-27
tags: [untyped_dependency_parsing, unlabelled_dependency_parsing, dependency_parsing, en, open_source]
supported: true
task: Dependency Parser
language: en
edition: Spark NLP 3.0.0
spark_version: 3.0
annotator: DependencyParserModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Untyped Dependency parser, trained on the on the CONLL dataset. 

Dependency parsing is the task of extracting a dependency parse of a sentence that represents its grammatical structure and defines the relationships between “head” words and words, which modify those heads.

Example:

root
|
| +-------dobj---------+
| |                    |
nsubj | |   +------det-----+ | +-----nmod------+
+--+  | |   |              | | |               |
|  |  | |   |      +-nmod-+| | |      +-case-+ |
+  |  + |   +      +      || + |      +      | |


I  prefer  the  morning   flight  through  Denver
Relations among the words are illustrated above the sentence with directed, labeled arcs from heads to dependents (+ indicates the dependent).

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dependency_conllu_en_3.0.0_3.0_1616860290925.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dependency_conllu_en_3.0.0_3.0_1616860290925.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
```scala

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val documentAssembler     = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector      = new SentenceDetector().setInputCols(Array("document")).setOutputCol("sentence")
val tokenizer             = new Tokenizer().setInputCols(Array("sentence")).setOutputCol("token")
val posTagger             = PerceptronModel.pretrained().setInputCols(Array("token", "sentence")).setOutputCol("pos")
val dependencyParser      = DependencyParserModel.pretrained().setInputCols(Array("sentence", "pos", "token")).setOutputCol("dependency")
val typedDependencyParser = TypedDependencyParserModel.pretrained().setInputCols(Array("token", "pos", "dependency")).setOutputCol("labdep")
val pipeline              = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, posTagger, dependencyParser, typedDependencyParser))
val df = Seq("Dependencies represents relationships betweens words in a Sentence").toDF("text")
val result = pipeline.fit(df).transform(df)
result.select("dependency.result").show(false)

```

{:.nlu-block}
```python
nlu.load("dep.untyped").predict("Dependencies represents relationships betweens words in a Sentence")
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------+
|result                                                                           |
+---------------------------------------------------------------------------------+
|[ROOT, Dependencies, represents, words, relationships, Sentence, Sentence, words]|
+---------------------------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dependency_conllu|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, pos, token]|
|Output Labels:|[dep_root]|
|Language:|en|

## Data Source

CONLL