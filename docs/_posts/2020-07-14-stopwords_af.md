---
layout: model
title: Stop Words Cleaner for Anglo-French
author: John Snow Labs
name: stopwords_af
date: 2020-07-14 19:03:00 +0800
tags: [stopwords, af]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_af_af_2.5.4_2.4_1594742440083.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
stop_words = StopWordsCleaner.pretrained("stopwords_af", "af") \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Anders as die koning van die noorde, is John Snow 'n Engelse dokter en 'n leier in die ontwikkeling van narkose en mediese higiëne.")
```

```scala
...
val stopWords = StopWordsCleaner.pretrained("stopwords_af", "af")
        .setInputCols(Array("token"))
        .setOutputCol("cleanTokens")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, stopWords))
val result = pipeline.fit(Seq.empty["Anders as die koning van die noorde, is John Snow 'n Engelse dokter en 'n leier in die ontwikkeling van narkose en mediese higiëne."].toDS.toDF("text")).transform(data)
```

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=0, end=5, result='Anders', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=14, end=19, result='koning', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=29, end=34, result='noorde', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=35, end=35, result=',', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=40, end=43, result='John', metadata={'sentence': '0'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_af|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|af|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)