---
layout: model
title: Sentiment Analysis (Vivekn)
author: John Snow Labs
name: sentiment_vivekn
date: 2021-11-22
tags: [sentiment, open_source, english, en, vivekn]
task: Sentiment Analysis
language: en
edition: Spark NLP 2.0.2
spark_version: 2.4
supported: true
annotator: ViveknSentimentModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model uses classifies text into negative and positive categories. It is based
on the approach by Vivek Narayanan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_vivekn_en_2.0.2_2.4_1556663184035.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentiment_vivekn_en_2.0.2_2.4_1556663184035.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

token = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")

normalizer = Normalizer() \
.setInputCols(["token"]) \
.setOutputCol("normal")

vivekn =  ViveknSentimentModel.pretrained() \
.setInputCols(["document", "normal"]) \
.setOutputCol("result_sentiment")

finisher = Finisher() \
.setInputCols(["result_sentiment"]) \
.setOutputCols("final_sentiment")

pipeline = Pipeline().setStages([document, token, normalizer, vivekn, finisher])

data = spark.createDataFrame([
["I recommend this movie"],
["Dont waste your time!!!"]
]).toDF("text")
pipelineModel = pipeline.fit(data)
result = pipelineModel.transform(data)

result.select("final_sentiment").show(truncate=False)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.Normalizer
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel
import com.johnsnowlabs.nlp.Finisher
import org.apache.spark.ml.Pipeline

val document = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val token = new Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val normalizer = new Normalizer()
.setInputCols("token")
.setOutputCol("normal")

val vivekn = ViveknSentimentModel.pretrained()
.setInputCols("document", "normal")
.setOutputCol("result_sentiment")

val finisher = new Finisher()
.setInputCols("result_sentiment")
.setOutputCols("final_sentiment")

val pipeline = new Pipeline().setStages(Array(document, token, normalizer, vivekn, finisher))

val data = Seq(
"I recommend this movie",
"Dont waste your time!!!"
).toDF("text")

val pipelineModel = pipeline.fit(data)

val result = pipelineModel.transform(data)
result.select("final_sentiment").show(false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.sentiment.vivekn").predict("""Dont waste your time!!!""")
```

</div>

## Results

```bash
+---------------+
|final_sentiment|
+---------------+
|[positive]     |
|[negative]     |
+---------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentiment_vivekn|
|Compatibility:|Spark NLP 2.0.2+|
|Edition:|Official|
|Input labels:|[document, token]|
|Output labels:|[sentiment]|
|Language:|en|
|License:|Open Source|


## Data Source

[AntBNC](https://www.laurenceanthony.net/software/antconc/), an automatically generated English lemma list based on all words in the BNC corpus with a frequency greater than 2 (created by Laurence Anthony)