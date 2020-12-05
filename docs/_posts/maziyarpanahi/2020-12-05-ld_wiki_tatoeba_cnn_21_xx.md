---
layout: model
title: Fast and Accurate Language Identification - 21 Languages
author: John Snow Labs
name: ld_wiki_tatoeba_cnn_21
date: 2020-12-05
tags: [open_source, language_detection, xx]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Language detection and identification is the task of automatically detecting the language(s) present in a document based on the content of the document. LanguageDetectorDL is an annotator that detects the language of documents or sentences depending on the inputCols. In addition, LanguageDetetorDL can accurately detect language from documents with mixed languages by coalescing sentences and select the best candidate.


We have designed and developed Deep Learning models by using CNNs and BiGRU architectures (mentioned in the model's name) in TensorFlow/Keras. The models are trained on large datasets such as Wikipedia and Tatoeba with high accuracy evaluated on the Europarl dataset. The output is a language code in Wiki Code style: https://en.wikipedia.org/wiki/List_of_Wikipedias

This model can detect the following languages:

\[`Bulgarian`, `Czech`, `Danish`, `German`, `Greek`, `English`, `Estonian`, `Finnish`, `French`, `Hungarian`, `Italian`, `Lithuanian`, `Latvian`, `Dutch`, `Polish`, `Portuguese`, `Romanian`, `Slovak`, `Slovenian`, `Spanish`, `Swedish`]

## Predicted Entities

\[`bg`, `cs`, `da`, `de`, `el`, `en`, `et`, `fi`, `fr`, `hu`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `sk`, `sl`, `es`, `sv`]

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_tatoeba_cnn_21_xx_2.7.0_2.4_1607177877570.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentence = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl", "xx")\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21")\
.setInputCols(["sentence"])\
.setOutputCol("language")\

languagePipeline = Pipeline(stages=[
 documentAssembler,
 sentence
 language_detector
])
```
```scala
val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentence = SentenceDetectorDLModel
  .pretrained("sentence_detector_dl", "xx")
  .setInputCols(Array("document"))
  .setOutputCol("sentence")

val languageDetector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21")
  .setInputCols("sentence")
  .setOutputCol("language")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentence,
    languageDetector
  ))
```
</div>

## Results

```bash
The result is a detected language code from `Predicted Entities`. For instance `en` as the English language.
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

Wikipedia and Tatoeba

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
