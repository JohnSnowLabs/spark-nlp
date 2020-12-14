---
layout: model
title: Fast and Accurate Language Identification - 95 Languages
author: John Snow Labs
name: ld_wiki_tatoeba_cnn_95
date: 2020-12-05
tags: [language_detection, open_source, xx]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Language detection and identification is the task of automatically detecting the language(s) present in a document based on the content of the document. LanguageDetectorDL is an annotator that detects the language of documents or sentences depending on the inputCols. In addition, LanguageDetetorDL can accurately detect language from documents with mixed languages by coalescing sentences and select the best candidate.


We have designed and developed Deep Learning models by using CNNs and BiGRU architectures (mentioned in the model's name) in TensorFlow/Keras. The models are trained on large datasets such as Wikipedia and Tatoeba with high accuracy evaluated on the Europarl dataset. The output is a language code in Wiki Code style: [https://en.wikipedia.org/wiki/List_of_Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias)

This model can detect the following languages:

\[`Afrikaans`, `Amharic`, `Aragonese`, `Arabic`, `Assamese`, `Azerbaijani`, `Belarusian`, `Bulgarian`, `Bengali`, `Breton`, `Bosnian`, `Catalan`, `Czech`, `Welsh`, `Danish`, `German`, `Greek`, `English`, `Esperanto`, `Spanish`, `Estonian`, `Basque`, `Persian`, `Finnish`, `Faroese`, `French`, `Irish`, `Galician`, `Gujarati`, `Hebrew`, `Hindi`, `Croatian`, `Haitian Creole`, `Hungarian`, `Armenian`, `Interlingua`, `Indonesian`, `Icelandic`, `Italian`, `Japanese`, `Javanese`, `Georgian`, `Kazakh`, `Khmer`, `Kannada`, `Korean`, `Kurdish`, `Kyrgyz`, `Latin`, `Luxembourgish`, `Lao`, `Lithuanian`, `Latvian`, `Malagasy`, `Macedonian`, `Malayalam`, `Mongolian`, `Marathi`, `Malay`, `Maltese`, `Nepali`, `Dutch`, `Norwegian Nynorsk`, `Norwegian`, `Occitan`, `Odia (Oriya)`, `Punjabi (Eastern)`, `Polish`, `Pashto`, `Portuguese`, `Quechua`, `Romanian`, `Russian`, `Northern Sami`, `Sinhala`, `Slovak`, `Slovenian`, `Albanian`, `Serbian`, `Swedish`, `Swahili`, `Tamil`, `Telugu`, `Thai`, `Tagalog`, `Turkish`, `Tatar`, `Uyghur`, `Ukrainian`, `Urdu`, `Vietnamese`, `Volapük`, `Walloon`, `Xhosa`, `Chinese`]

## Predicted Entities

\[`af`, `am`, `an`, `ar`, `as`, `az`, `be`, `bg`, `bn`, `br`, `bs`, `ca`, `cs`, `cy`, `da`, `de`, `el`, `en`, `eo`, `es`, `et`, `eu`, `fa`, `fi`, `fo`, `fr`, `ga`, `gl`, `gu`, `he`, `hi`, `hr`, `ht`, `hu`, `hy`, `ia`, `id`, `is`, `it`, `ja`, `jv`, `ka`, `kk`, `km`, `kn`, `ko`, `ku`, `ky`, `la`, `lb`, `lo`, `lt`, `lv`, `mg`, `mk`, `ml`, `mn`, `mr`, `ms`, `mt`, `ne`, `nl`, `nn`, `no`, `oc`, `or`, `pa`, `pl`, `ps`, `pt`, `qu`, `ro`, `ru`, `se`, `si`, `sk`, `sl`, `sq`, `sr`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`, `tt`, `ug`, `uk`, `ur`, `vi`, `vo`, `wa`, `xh`, `zh`]

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_tatoeba_cnn_95_xx_2.7.0_2.4_1607184332861.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_95")\
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

val languageDetector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_95")
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
|Model Name:|ld_wiki_tatoeba_cnn_95|
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
|      fr| 1000|    999|             0.999|
|      de| 1000|    998|             0.998|
|      fi| 1000|    998|             0.998|
|      pt| 1000|    996|             0.996|
|      sv| 1000|    995|             0.995|
|      el| 1000|    994|             0.994|
|      nl| 1000|    994|             0.994|
|      it| 1000|    994|             0.994|
|      en| 1000|    993|             0.993|
|      es| 1000|    984|             0.984|
|      hu|  880|    865|0.9829545454545454|
|      ro|  784|    769|0.9808673469387755|
|      lt| 1000|    978|             0.978|
|      et|  928|    906|0.9762931034482759|
|      cs| 1000|    975|             0.975|
|      pl|  914|    890| 0.973741794310722|
|      da| 1000|    958|             0.958|
|      sk| 1000|    947|             0.947|
|      bg| 1000|    939|             0.939|
|      lv|  916|    849|0.9268558951965066|
|      sl|  914|    844|0.9234135667396062|
+--------+-----+-------+------------------+

+-------+-------------------+
|summary|          precision|
+-------+-------------------+
|  count|                 21|
|   mean| 0.9764822024804014|
| stddev|0.02384830734809143|
|    min| 0.9234135667396062|
|    max|              0.999|
+-------+-------------------+
```