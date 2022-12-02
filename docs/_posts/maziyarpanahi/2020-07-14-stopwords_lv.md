---
layout: model
title: Stop Words Cleaner for Latvian
author: John Snow Labs
name: stopwords_lv
date: 2020-07-14 19:03:00 +0800
task: Stop Words Removal
language: lv
edition: Spark NLP 2.5.4
spark_version: 2.4
tags: [stopwords, lv]
supported: true
annotator: StopWordsCleaner
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This model removes 'stop words' from text. Stop words are words so common that they can be removed without significantly altering the meaning of a text. Removing stop words is useful when one wants to deal with only the most semantically important words in a text, and ignore words that are rarely semantically relevant, such as articles and prepositions.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/b2eb08610dd49d5b15077cc499a94b4ec1e8b861/jupyter/annotation/english/stop-words/StopWordsCleaner.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_lv_lv_2.5.4_2.4_1594742439893.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
stop_words = StopWordsCleaner.pretrained("stopwords_lv", "lv") \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Džons Snovs ir ne tikai ziemeļu karalis, bet arī angļu ārsts un anestēzijas un medicīniskās higiēnas attīstības līderis.")
```

```scala
...
val stopWords = StopWordsCleaner.pretrained("stopwords_lv", "lv")
        .setInputCols(Array("token"))
        .setOutputCol("cleanTokens")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, stopWords))
val data = Seq("Džons Snovs ir ne tikai ziemeļu karalis, bet arī angļu ārsts un anestēzijas un medicīniskās higiēnas attīstības līderis.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Džons Snovs ir ne tikai ziemeļu karalis, bet arī angļu ārsts un anestēzijas un medicīniskās higiēnas attīstības līderis."""]
stopword_df = nlu.load('lv.stopwords').predict(text)
stopword_df[['cleanTokens']]
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=0, end=4, result='Džons', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=6, end=10, result='Snovs', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=24, end=30, result='ziemeļu', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=32, end=38, result='karalis', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=39, end=39, result=',', metadata={'sentence': '0'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_lv|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|lv|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)