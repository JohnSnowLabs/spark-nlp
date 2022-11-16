---
layout: model
title: Stop Words Cleaner for Indonesian
author: John Snow Labs
name: stopwords_id
date: 2020-07-14 19:03:00 +0800
task: Stop Words Removal
language: id
edition: Spark NLP 2.5.4
spark_version: 2.4
tags: [stopwords, id]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_id_id_2.5.4_2.4_1594742441630.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
stop_words = StopWordsCleaner.pretrained("stopwords_id", "id") \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Selain menjadi raja utara, John Snow adalah seorang dokter Inggris dan pemimpin dalam pengembangan anestesi dan kebersihan medis.")
```

```scala
...
val stopWords = StopWordsCleaner.pretrained("stopwords_id", "id")
        .setInputCols(Array("token"))
        .setOutputCol("cleanTokens")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, stopWords))
val data = Seq("Selain menjadi raja utara, John Snow adalah seorang dokter Inggris dan pemimpin dalam pengembangan anestesi dan kebersihan medis.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Selain menjadi raja utara, John Snow adalah seorang dokter Inggris dan pemimpin dalam pengembangan anestesi dan kebersihan medis."""]
stopword_df = nlu.load('id.stopwords').predict(text)
stopword_df[['cleanTokens']]
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=7, end=13, result='menjadi', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=15, end=18, result='raja', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=20, end=24, result='utara', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=25, end=25, result=',', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=27, end=30, result='John', metadata={'sentence': '0'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_id|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|id|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)