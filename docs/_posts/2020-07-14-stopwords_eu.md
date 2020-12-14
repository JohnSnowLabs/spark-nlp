---
layout: model
title: Stop Words Cleaner for Basque
author: John Snow Labs
name: stopwords_eu
date: 2020-07-14 19:03:00 +0800
tags: [stopwords, eu]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_eu_eu_2.5.4_2.4_1594742441951.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
stop_words = StopWordsCleaner.pretrained("stopwords_eu", "eu") \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Iparraldeko erregea izateaz gain, mediku ingelesa eta anestesia eta higiene medikoa garatzen duen liderra da John Snow.")
```

```scala
...
val stopWords = StopWordsCleaner.pretrained("stopwords_eu", "eu")
        .setInputCols(Array("token"))
        .setOutputCol("cleanTokens")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, stopWords))
val result = pipeline.fit(Seq.empty["Iparraldeko erregea izateaz gain, mediku ingelesa eta anestesia eta higiene medikoa garatzen duen liderra da John Snow."].toDS.toDF("text")).transform(data)
```

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=0, end=10, result='Iparraldeko', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=12, end=18, result='erregea', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=20, end=26, result='izateaz', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=28, end=31, result='gain', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=32, end=32, result=',', metadata={'sentence': '0'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_eu|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|eu|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)