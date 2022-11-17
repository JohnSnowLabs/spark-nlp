---
layout: model
title: Stop Words Cleaner for Zulu
author: John Snow Labs
name: stopwords_zu
date: 2020-07-14 19:03:00 +0800
task: Stop Words Removal
language: zu
edition: Spark NLP 2.5.4
spark_version: 2.4
tags: [stopwords, zu]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_zu_zu_2.5.4_2.4_1594742440504.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
stop_words = StopWordsCleaner.pretrained("stopwords_zu", "zu") \
.setInputCols(["token"]) \
.setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Ngaphandle kokuba yinkosi yasenyakatho, uJohn Snow ungudokotela waseNgilandi futhi ungumholi ekwenziweni kwe-anesthesia kanye nenhlanzeko yezokwelapha.")
```

```scala
...
val stopWords = StopWordsCleaner.pretrained("stopwords_zu", "zu")
.setInputCols(Array("token"))
.setOutputCol("cleanTokens")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, stopWords))
val data = Seq("Ngaphandle kokuba yinkosi yasenyakatho, uJohn Snow ungudokotela waseNgilandi futhi ungumholi ekwenziweni kwe-anesthesia kanye nenhlanzeko yezokwelapha.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Ngaphandle kokuba yinkosi yasenyakatho, uJohn Snow ungudokotela waseNgilandi futhi ungumholi ekwenziweni kwe-anesthesia kanye nenhlanzeko yezokwelapha."""]
stopword_df = nlu.load('zu.stopwords').predict(text)
stopword_df[['cleanTokens']]
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=0, end=9, result='Ngaphandle', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=11, end=16, result='kokuba', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=18, end=24, result='yinkosi', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=26, end=37, result='yasenyakatho', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=38, end=38, result=',', metadata={'sentence': '0'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_zu|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|zu|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)