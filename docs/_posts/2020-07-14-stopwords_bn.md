---
layout: model
title: Stop Words Cleaner for Bengali
author: John Snow Labs
name: stopwords_bn
date: 2020-07-14 19:03:00 +0800
tags: [stopwords, bn]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model removes 'stop words' from text. Stop words are words so common that they can removed without significantly altering the meaning of a text. Removing stop words is useful when one wants to deal with only the most semantically important words in a text, and ignore words that are rarely semantically relevant, such as articles and prepositions.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/b2eb08610dd49d5b15077cc499a94b4ec1e8b861/jupyter/annotation/english/stop-words/StopWordsCleaner.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_bn_bn_2.5.4_2.4_1594742440339.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

stop_words = StopWordsCleaner.pretrained("stopwords_bn", "bn") \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("উত্তরের রাজা হওয়া ছাড়াও জন স্নো একজন ইংরেজ চিকিত্সক এবং অবেদন এবং মেডিকেল হাইজিনের বিকাশের এক নেতা is")
```

```scala

val stopWords = StopWordsCleaner.pretrained("stopwords_bn", "bn")
        .setInputCols(Array("token"))
        .setOutputCol("cleanTokens")
```

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=0, end=6, result='উত্তরের', metadata={'sentence': '0'}, embeddings=[]),
Row(annotatorType='token', begin=8, end=11, result='রাজা', metadata={'sentence': '0'}, embeddings=[]),
Row(annotatorType='token', begin=13, end=17, result='হওয়া', metadata={'sentence': '0'}, embeddings=[]),
Row(annotatorType='token', begin=19, end=24, result='ছাড়াও', metadata={'sentence': '0'}, embeddings=[]),
Row(annotatorType='token', begin=26, end=27, result='জন', metadata={'sentence': '0'}, embeddings=[]),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_bn|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|bn|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)