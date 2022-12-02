---
layout: model
title: English Lemmatizer
author: John Snow Labs
name: lemma_antbnc
date: 2021-11-22
tags: [lemmatizer, open_source, english, en]
task: Lemmatization
language: en
edition: Spark NLP 2.0.2
spark_version: 2.4
supported: true
annotator: LemmatizerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model uses context and language knowledge to assign all forms and inflections of a word to a single root. This enables the pipeline to treat the past and present tense of a verb, for example, as the same word instead of two completely different words. The lemmatizer takes into consideration the context surrounding a word to determine which root is correct when the word form alone is ambiguous.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemma_antbnc_en_2.0.2_2.4_1556480454569.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentenceDetector = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = Tokenizer() \
.setInputCols(["sentence"]) \
.setOutputCol("token")

lemmatizer = LemmatizerModel.pretrained() \
.setInputCols(["token"]) \
.setOutputCol("lemma")

pipeline = Pipeline() \
.setStages([
documentAssembler,
sentenceDetector,
tokenizer,
lemmatizer
])

data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers."]]) \
.toDF("text")

result = pipeline.fit(data).transform(data)
result.selectExpr("lemma.result").show(truncate=False)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Lemmatizer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = new SentenceDetector()
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val lemmatizer = new Lemmatizer()
.setInputCols(Array("token"))
.setOutputCol("lemma")
.setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")

val pipeline = new Pipeline()
.setStages(Array(
documentAssembler,
sentenceDetector,
tokenizer,
lemmatizer
))

val data = Seq("Peter Pipers employees are picking pecks of pickled peppers.")
.toDF("text")

val result = pipeline.fit(data).transform(data)
result.selectExpr("lemma.result").show(false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.lemma.antbnc").predict("""Peter Pipers employees are picking pecks of pickled peppers.""")
```

</div>

## Results

```bash
+------------------------------------------------------------------+
|result                                                            |
+------------------------------------------------------------------+
|[Peter, Pipers, employees, are, pick, peck, of, pickle, pepper, .]|
+------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|lemma_antbnc|
|Type:|lemmatizer|
|Compatibility:|Spark NLP 2.4.0+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[lemma]|
|Language:|en|
|Case sensitive:|false|
|License:|Open Source|


## Data Source

[AntBNC](https://www.laurenceanthony.net/software/antconc/), an automatically generated English lemma list based on all words in the BNC corpus with a frequency greater than 2 (created by Laurence Anthony)
