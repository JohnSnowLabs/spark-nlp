---
layout: model
title: Fast and Accurate Language Identification - 21 Languages (CNN)
author: John Snow Labs
name: ld_wiki_tatoeba_cnn_21
date: 2020-12-05
task: Language Detection
language: xx
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [open_source, language_detection, xx]
supported: true
annotator: LanguageDetectorDL
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Language detection and identification is the task of automatically detecting the language(s) present in a document based on the content of the document. ``LanguageDetectorDL`` is an annotator that detects the language of documents or sentences depending on the ``inputCols``. In addition, ``LanguageDetetorDL`` can accurately detect language from documents with mixed languages by coalescing sentences and select the best candidate.

We have designed and developed Deep Learning models using CNNs in TensorFlow/Keras. The model is trained on large datasets such as Wikipedia and Tatoeba with high accuracy evaluated on the Europarl dataset. The output is a language code in Wiki Code style: [https://en.wikipedia.org/wiki/List_of_Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias).

This model can detect the following languages:

`Bulgarian`, `Czech`, `Danish`, `German`, `Greek`, `English`, `Estonian`, `Finnish`, `French`, `Hungarian`, `Italian`, `Lithuanian`, `Latvian`, `Dutch`, `Polish`, `Portuguese`, `Romanian`, `Slovak`, `Slovenian`, `Spanish`, `Swedish`.

## Predicted Entities

`bg`, `cs`, `da`, `de`, `el`, `en`, `et`, `fi`, `fr`, `hu`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `sk`, `sl`, `es`, `sv`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_tatoeba_cnn_21_xx_2.7.0_2.4_1607177877570.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ld_wiki_tatoeba_cnn_21_xx_2.7.0_2.4_1607177877570.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21", "xx")\
.setInputCols(["sentence"])\
.setOutputCol("language")
languagePipeline = Pipeline(stages=[documentAssembler, sentenceDetector, language_detector])
light_pipeline = LightPipeline(languagePipeline.fit(spark.createDataFrame([['']]).toDF("text")))
result = light_pipeline.fullAnnotate("Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.")
```
```scala
...
val languageDetector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21", "xx")
.setInputCols("sentence")
.setOutputCol("language")
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, languageDetector))
val data = Seq("Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala."]
lang_df = nlu.load('xx.classify.wiki_21').predict(text, output_level='sentence')
lang_df
```

</div>

## Results

```bash
'fr'
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ld_wiki_tatoeba_cnn_21|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[language]|
|Language:|xx|

## Data Source

Wikipedia and Tatoeba.

## Benchmarking

```bash
Evaluated on Europarl dataset which the model has never seen:

+--------+-----+-------+------------------+
|src_lang|count|correct|         precision|
+--------+-----+-------+------------------+
|      de| 1000|   1000|               1.0|
|      nl| 1000|   1000|               1.0|
|      pt| 1000|   1000|               1.0|
|      fr| 1000|   1000|               1.0|
|      es| 1000|   1000|               1.0|
|      it| 1000|   1000|               1.0|
|      fi| 1000|   1000|               1.0|
|      da| 1000|    999|             0.999|
|      en| 1000|    999|             0.999|
|      sv| 1000|    998|             0.998|
|      el| 1000|    996|             0.996|
|      bg| 1000|    989|             0.989|
|      pl|  914|    903|0.9879649890590809|
|      hu|  880|    867|0.9852272727272727|
|      ro|  784|    771|0.9834183673469388|
|      lt| 1000|    982|             0.982|
|      sk| 1000|    976|             0.976|
|      et|  928|    903|0.9730603448275862|
|      cs| 1000|    967|             0.967|
|      sl|  914|    875|0.9573304157549234|
|      lv|  916|    869|0.9486899563318777|
+--------+-----+-------+------------------+

+-------+--------------------+
|summary|           precision|
+-------+--------------------+
|  count|                  21|
|   mean|  0.9876995879070323|
| stddev|0.015446490915012105|
|    min|  0.9486899563318777|
|    max|                 1.0|
+-------+--------------------+
```
