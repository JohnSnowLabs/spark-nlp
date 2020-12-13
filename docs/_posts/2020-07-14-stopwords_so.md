---
layout: model
title: Stop Words Cleaner for Somali
author: John Snow Labs
name: stopwords_so
date: 2020-07-14 19:03:00 +0800
tags: [stopwords, so]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model removes 'stop words' from text. Stop words are words so common that they can removed without significantly altering the meaning of a text. Removing stop words is useful when one wants to deal with only the most semantically important words in a text, and ignore words that are rarely semantically relevant, such as articles and prepositions.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/b2eb08610dd49d5b15077cc499a94b4ec1e8b861/jupyter/annotation/english/stop-words/StopWordsCleaner.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_so_so_2.5.4_2.4_1594742441799.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

stop_words = StopWordsCleaner.pretrained("stopwords_so", "so") \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Marka laga reebo inuu yahay boqorka woqooyiga, John Snow waa dhakhtar Ingiriis ah oo hormuud u ah horumarinta suuxdinta iyo nadaafadda caafimaadka.")
```

```scala

val stopWords = StopWordsCleaner.pretrained("stopwords_so", "so")
        .setInputCols(Array("token"))
        .setOutputCol("cleanTokens")
```

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=0, end=4, result='Marka', metadata={'sentence': '0'}, embeddings=[]),
Row(annotatorType='token', begin=6, end=9, result='laga', metadata={'sentence': '0'}, embeddings=[]),
Row(annotatorType='token', begin=11, end=15, result='reebo', metadata={'sentence': '0'}, embeddings=[]),
Row(annotatorType='token', begin=22, end=26, result='yahay', metadata={'sentence': '0'}, embeddings=[]),
Row(annotatorType='token', begin=28, end=34, result='boqorka', metadata={'sentence': '0'}, embeddings=[]),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_so|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|so|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)