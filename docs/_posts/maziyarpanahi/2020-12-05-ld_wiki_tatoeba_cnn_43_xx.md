---
layout: model
title: Fast and Accurate Language Identification - 43 Languages (CNN)
author: John Snow Labs
name: ld_wiki_tatoeba_cnn_43
date: 2020-12-05
task: Language Detection
language: xx
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [language_detection, open_source, xx]
supported: true
annotator: LanguageDetectorDL
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Language detection and identification is the task of automatically detecting the language(s) present in a document based on the content of the document. ``LanguageDetectorDL`` is an annotator that detects the language of documents or sentences depending on the ``inputCols``. In addition, ``LanguageDetetorDL`` can accurately detect language from documents with mixed languages by coalescing sentences and select the best candidate.

We have designed and developed Deep Learning models using CNNs in TensorFlow/Keras. The models are trained on large datasets such as Wikipedia and Tatoeba with high accuracy evaluated on the Europarl dataset. The output is a language code in Wiki Code style: [https://en.wikipedia.org/wiki/List_of_Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias).

This model can detect the following languages:

`Arabic`, `Belarusian`, `Bulgarian`, `Czech`, `Danish`, `German`, `Greek`, `English`, `Esperanto`, `Spanish`, `Estonian`, `Persian`, `Finnish`, `French`, `Hebrew`, `Hindi`, `Hungarian`, `Interlingua`, `Indonesian`, `Icelandic`, `Italian`, `Japanese`, `Korean`, `Latin`, `Lithuanian`, `Latvian`, `Macedonian`, `Marathi`, `Dutch`, `Polish`, `Portuguese`, `Romanian`, `Russian`, `Slovak`, `Slovenian`, `Serbian`, `Swedish`, `Tagalog`, `Turkish`, `Tatar`, `Ukrainian`, `Vietnamese`, `Chinese`.

## Predicted Entities

`ar`, `be`, `bg`, `cs`, `da`, `de`, `el`, `en`, `eo`, `es`, `et`, `fa`, `fi`, `fr`, `he`, `hi`, `hu`, `ia`, `id`, `is`, `it`, `ja`, `ko`, `la`, `lt`, `lv`, `mk`, `mr`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sr`, `sv`, `tl`, `tr`, `tt`, `uk`, `vi`, `zh`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_tatoeba_cnn_43_xx_2.7.0_2.4_1607184003726.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ld_wiki_tatoeba_cnn_43_xx_2.7.0_2.4_1607184003726.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_43", "xx")\
.setInputCols(["sentence"])\
.setOutputCol("language")
languagePipeline = Pipeline(stages=[documentAssembler, sentenceDetector, language_detector])
light_pipeline = LightPipeline(languagePipeline.fit(spark.createDataFrame([['']]).toDF("text")))
result = light_pipeline.fullAnnotate("Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.")
```
```scala
...
val languageDetector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_43", "xx")
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
lang_df = nlu.load('xx.classify.wiki_43').predict(text, output_level='sentence')
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
|Model Name:|ld_wiki_tatoeba_cnn_43|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[language]|
|Language:|xx|

## Data Source

Wikipedia and Tatoeba

## Benchmarking

```bash
Evaluated on Europarl dataset which the model has never seen:

+--------+-----+-------+------------------+
|src_lang|count|correct|         precision|
+--------+-----+-------+------------------+
|      fr| 1000|   1000|               1.0|
|      nl| 1000|    999|             0.999|
|      sv| 1000|    999|             0.999|
|      pt| 1000|    999|             0.999|
|      it| 1000|    999|             0.999|
|      es| 1000|    999|             0.999|
|      fi| 1000|    999|             0.999|
|      el| 1000|    998|             0.998|
|      de| 1000|    997|             0.997|
|      da| 1000|    997|             0.997|
|      en| 1000|    995|             0.995|
|      lt| 1000|    986|             0.986|
|      hu|  880|    867|0.9852272727272727|
|      pl|  914|    899|0.9835886214442013|
|      ro|  784|    765|0.9757653061224489|
|      et|  928|    899|           0.96875|
|      cs| 1000|    967|             0.967|
|      sk| 1000|    966|             0.966|
|      bg| 1000|    960|              0.96|
|      sl|  914|    860|0.9409190371991247|
|      lv|  916|    856|0.9344978165938864|
+--------+-----+-------+------------------+

+-------+--------------------+
|summary|           precision|
+-------+--------------------+
|  count|                  21|
|   mean|  0.9832737168612825|
| stddev|0.020064155103808722|
|    min|  0.9344978165938864|
|    max|                 1.0|
+-------+--------------------+
```