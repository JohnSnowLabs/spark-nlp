---
layout: model
title: Stop Words Cleaner for Polish
author: John Snow Labs
name: stopwords_pl
date: 2020-07-14 19:03:00 +0800
tags: [stopwords, pl]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This model removes 'stop words' from text. Stop words are words so common that they can be removed without significantly altering the meaning of a text. Removing stop words is useful when one wants to deal with only the most semantically important words in a text, and ignore words that are rarely semantically relevant, such as articles and prepositions.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/b2eb08610dd49d5b15077cc499a94b4ec1e8b861/jupyter/annotation/english/stop-words/StopWordsCleaner.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_pl_pl_2.5.4_2.4_1594742438519.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
stop_words = StopWordsCleaner.pretrained("stopwords_pl", "pl") \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Oprócz bycia królem północy, John Snow jest angielskim lekarzem i liderem w rozwoju anestezjologii i higieny medycznej.")
```

```scala
...
val stopWords = StopWordsCleaner.pretrained("stopwords_pl", "pl")
        .setInputCols(Array("token"))
        .setOutputCol("cleanTokens")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, stopWords))
val result = pipeline.fit(Seq.empty["Oprócz bycia królem północy, John Snow jest angielskim lekarzem i liderem w rozwoju anestezjologii i higieny medycznej."].toDS.toDF("text")).transform(data)
```

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=0, end=5, result='Oprócz', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=7, end=11, result='bycia', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=13, end=18, result='królem', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=20, end=26, result='północy', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=27, end=27, result=',', metadata={'sentence': '0'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_pl|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|pl|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)