---
layout: model
title: Fast and Accurate Language Identification - 99 Languages (CNN)
author: John Snow Labs
name: ld_tatoeba_cnn_99
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

We have designed and developed Deep Learning models using CNNs in TensorFlow/Keras. The model is trained on Tatoeba dataset with high accuracy evaluated on the Europarl dataset. The output is a language code in Wiki Code style: [https://en.wikipedia.org/wiki/List_of_Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias).

This model can detect the following languages:

`Afrikaans`, `Arabic`, `Algerian Arabic`, `Assamese`, `Kotava`, `Azerbaijani`, `Belarusian`, `Bengali`, `Berber`, `Breton`, `Bulgarian`, `Catalan`, `Chavacano`, `Cebuano`, `Czech`, `Chuvash`, `Mandarin Chinese`, `Cornish`, `Danish`, `German`, `Central Dusun`, `Modern Greek (1453-)`, `English`, `Esperanto`, `Estonian`, `Basque`, `Finnish`, `French`, `Guadeloupean Creole French`, `Irish`, `Galician`, `Gronings`, `Guarani`, `Hebrew`, `Hindi`, `Croatian`, `Hungarian`, `Armenian`, `Ido`, `Interlingue`, `Ilocano`, `Interlingua`, `Indonesian`, `Icelandic`, `Italian`, `Lojban`, `Japanese`, `Kabyle`, `Georgian`, `Kazakh`, `Khasi`, `Khmer`, `Korean`, `Coastal Kadazan`, `Latin`, `Lingua Franca Nova`, `Lithuanian`, `Latvian`, `Literary Chinese`, `Marathi`, `Meadow Mari`, `Macedonian`, `Low German (Low Saxon)`, `Dutch`, `Norwegian Nynorsk`, `Norwegian Bokmål`, `Occitan`, `Ottoman Turkish`, `Kapampangan`, `Picard`, `Persian`, `Polish`, `Portuguese`, `Romanian`, `Kirundi`, `Russian`, `Slovak`, `Spanish`, `Albanian`, `Serbian`, `Swedish`, `Swabian`, `Tatar`, `Tagalog`, `Thai`, `Klingon`, `Toki Pona`, `Turkmen`, `Turkish`, `Uyghur`, `Ukrainian`, `Urdu`, `Vietnamese`, `Volapük`, `Waray`, `Shanghainese`, `Yiddish`, `Cantonese`, `Malay`.

## Predicted Entities

`af`, `ar`, `arq`, `as`, `avk`, `az`, `be`, `bn`, `ber`, `br`, `bg`, `ca`, `cbk`, `ceb`, `cs`, `cv`, `cmn`, `kw`, `da`, `de`, `dtp`, `el`, `en`, `eo`, `et`, `eu`, `fi`, `fr`, `gcf`, `ga`, `gl`, `gos`, `gn`, `he`, `hi`, `hr`, `hu`, `hy`, `io`, `ie`, `ilo`, `ia`, `id`, `is`, `it`, `jbo`, `ja`, `kab`, `ka`, `kk`, `kha`, `km`, `ko`, `kzj`, `la`, `lfn`, `lt`, `lvs`, `lzh`, `mr`, `mhr`, `mk`, `nds`, `nl`, `nn`, `nb`, `oc`, `ota`, `pam`, `pcd`, `pes`, `pl`, `pt`, `ro`, `rn`, `ru`, `sk`, `es`, `sq`, `sr`, `sv`, `swg`, `tt`, `tl`, `th`, `tlh`, `toki`, `tk`, `tr`, `ug`, `uk`, `ur`, `vi`, `vo`, `war`, `wuu`, `yi`, `yue`, `zsm`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_tatoeba_cnn_99_xx_2.7.0_2.4_1607183215533.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
language_detector = LanguageDetectorDL.pretrained("ld_tatoeba_cnn_99", "xx")\
.setInputCols(["sentence"])\
.setOutputCol("language")
languagePipeline = Pipeline(stages=[documentAssembler, sentenceDetector, language_detector])
light_pipeline = LightPipeline(languagePipeline.fit(spark.createDataFrame([['']]).toDF("text")))
result = light_pipeline.fullAnnotate("Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.")
```
```scala
...
val languageDetector = LanguageDetectorDL.pretrained("ld_tatoeba_cnn_99", "xx")
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
lang_df = nlu.load('xx.classify.wiki_99').predict(text, output_level='sentence')
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
|Model Name:|ld_tatoeba_cnn_99|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[language]|
|Language:|xx|

## Data Source

Tatoeba

## Benchmarking

```bash
Evaluated on Europarl dataset which the model has never seen:

+--------+-----+-------+------------------+
|src_lang|count|correct|         precision|
+--------+-----+-------+------------------+
|      fi| 1000|   1000|               1.0|
|      pt| 1000|   1000|               1.0|
|      it| 1000|    998|             0.998|
|      es| 1000|    997|             0.997|
|      en| 1000|    995|             0.995|
|      da| 1000|    994|             0.994|
|      sv| 1000|    992|             0.992|
|      pl|  914|    899|0.9835886214442013|
|      hu|  880|    863|0.9806818181818182|
|      lt| 1000|    975|             0.975|
|      bg| 1000|    951|             0.951|
|      et|  928|    783|           0.84375|
+--------+-----+-------+------------------+

+-------+-------------------+
|summary|          precision|
+-------+-------------------+
|  count|                 12|
|   mean| 0.9758350366355016|
| stddev|0.04391442353856736|
|    min|            0.84375|
|    max|                1.0|
+-------+-------------------+
```
