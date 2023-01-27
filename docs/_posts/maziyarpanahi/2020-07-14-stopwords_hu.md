---
layout: model
title: Stop Words Cleaner for Hungarian
author: John Snow Labs
name: stopwords_hu
date: 2020-07-14 19:03:00 +0800
task: Stop Words Removal
language: hu
edition: Spark NLP 2.5.4
spark_version: 2.4
tags: [stopwords, hu]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stopwords_hu_hu_2.5.4_2.4_1594742441137.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/stopwords_hu_hu_2.5.4_2.4_1594742441137.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
stop_words = StopWordsCleaner.pretrained("stopwords_hu", "hu") \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens")
nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, stop_words])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Az északi király kivételével John Snow angol orvos, vezető szerepet játszik az érzéstelenítés és az orvosi higiénia fejlesztésében.")
```

```scala
...
val stopWords = StopWordsCleaner.pretrained("stopwords_hu", "hu")
        .setInputCols(Array("token"))
        .setOutputCol("cleanTokens")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, stopWords))
val data = Seq("Az északi király kivételével John Snow angol orvos, vezető szerepet játszik az érzéstelenítés és az orvosi higiénia fejlesztésében.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Az északi király kivételével John Snow angol orvos, vezető szerepet játszik az érzéstelenítés és az orvosi higiénia fejlesztésében."""]
stopword_df = nlu.load('hu.stopwords').predict(text)
stopword_df[['cleanTokens']]
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='token', begin=3, end=8, result='északi', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=10, end=15, result='király', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=17, end=27, result='kivételével', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=29, end=32, result='John', metadata={'sentence': '0'}),
Row(annotatorType='token', begin=34, end=37, result='Snow', metadata={'sentence': '0'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stopwords_hu|
|Type:|stopwords|
|Compatibility:|Spark NLP 2.5.4+|
|Edition:|Official|
|Input Labels:|[token]|
|Output Labels:|[cleanTokens]|
|Language:|hu|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/WorldBrain/remove-stopwords](https://github.com/WorldBrain/remove-stopwords)