---
layout: model
title: My Pipeline for News Type classification
author: John Snow Labs
name: my_nlp_pipeline
date: 2021-10-12
tags: [en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was trained on alALBALBLABLAB

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/my_nlp_pipeline_en_3.3.1_3.0_1634068169120.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
result.select("dependency.result", "labdep.result").show(false)

```

{:.nlu-block}
```python

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
result.select("dependency.result", "labdep.result").show(false)

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
|Model Name:|my_nlp_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.3.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Dependencies:|BertSentenceEmbeddings|

## Included Models

- DocumentAssembler
- TokenizerModel
- NormalizerModel
- StopWordsCleaner
- LemmatizerModel
- WordEmbeddingsModel
- SentenceEmbeddings
- ClassifierDLModel