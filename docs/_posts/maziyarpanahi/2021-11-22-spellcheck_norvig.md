---
layout: model
title: Spell Checker for the English Language (Norvig)
author: John Snow Labs
name: spellcheck_norvig
date: 2021-11-22
tags: [en, open_source, spell_check, norvig]
supported: true
task: Spell Check
language: en
edition: Spark NLP 2.0.2
spark_version: 2.4
annotator: NorvigSweetingModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Detects and corrects spelling errors in your input text. It’s based on the NorvigSweeting approach.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/spellcheck_norvig_en_2.0.2_2.4_1556605026653.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/spellcheck_norvig_en_2.0.2_2.4_1556605026653.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

tokenizer = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")

spellChecker = NorvigSweetingModel.pretrained() \
.setInputCols(["token"]) \
.setOutputCol("spell")

pipeline = Pipeline().setStages([
documentAssembler,
tokenizer,
spellChecker
])

data = spark.createDataFrame([["somtimes i wrrite wordz erong."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("spell.result").show(truncate=False)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel

import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val spellChecker = NorvigSweetingModel.pretrained()
.setInputCols("token")
.setOutputCol("spell")

val pipeline = new Pipeline().setStages(Array(
documentAssembler,
tokenizer,
spellChecker
))

val data = Seq("somtimes i wrrite wordz erong.").toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("spell.result").show(false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.spell.norvig").predict("""somtimes i wrrite wordz erong.""")
```

</div>

## Results
```bash
+--------------------------------------+
|result                                |
+--------------------------------------+
|[sometimes, i, write, words, wrong, .]|
+--------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|spellcheck_norvig|
|Compatibility:|Spark NLP 2.0.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[checked]|
|Language:|en|
