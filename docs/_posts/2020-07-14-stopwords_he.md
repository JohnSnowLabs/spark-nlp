---
layout: model
title: Stop Words Cleaner for Hebrew
author: John Snow Labs
name: stopwords_he
date: 2020-07-14 19:03:00 +0800
tags: [stopwords, he]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_he_he_2.5.4_2.4_1594742441877.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
stop_words = StopWordsCleaner.pretrained("stopwords_he", "he") \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("מלבד היותו מלך הצפון, ג'ון סנואו הוא רופא אנגלי ומוביל בפיתוח הרדמה והיגיינה רפואית.")
```

```scala
...
val stopWords = StopWordsCleaner.pretrained("stopwords_he", "he")
        .setInputCols(Array("token"))
        .setOutputCol("cleanTokens")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, stopWords))
val result = pipeline.fit(Seq.empty["מלבד היותו מלך הצפון, ג'ון סנואו הוא רופא אנגלי ומוביל בפיתוח הרדמה והיגיינה רפואית."].toDS.toDF("text")).transform(data)
```

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=5, end=9, result='היותו', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=11, end=13, result='מלך', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=15, end=19, result='הצפון', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=20, end=20, result=',', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=22, end=25, result="ג'ון", metadata={'sentence': '0'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_he|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|he|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)