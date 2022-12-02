---
layout: model
title: Fast and Accurate Language Identification - 220 Languages (CNN)
author: John Snow Labs
name: ld_wiki_tatoeba_cnn_220
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

We have designed and developed Deep Learning models using CNNs in TensorFlow/Keras. The model is trained on large datasets such as Wikipedia and Tatoeba with high accuracy evaluated on the Europarl dataset. The output is a language code in Wiki Code style: [https://en.wikipedia.org/wiki/List_of_Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias).

This model can detect the following languages:

`Achinese`, `Afrikaans`, `Tosk Albanian`, `Amharic`, `Aragonese`, `Old English`, `Arabic`, `Egyptian Arabic`, `Assamese`, `Asturian`, `Avaric`, `Aymara`, `Azerbaijani`, `South Azerbaijani`, `Bashkir`, `Bavarian`, `bat-smg`, `Central Bikol`, `Belarusian`, `Bulgarian`, `bh`, `Bengali`, `Tibetan`, `Bishnupriya`, `Breton`, `Russia Buriat`, `Catalan`, `Min Dong Chinese`, `Chechen`, `Cebuano`, `Central Kurdish (Soranî)`, `Corsican`, `Crimean Tatar`, `Czech`, `Kashubian`, `Chuvash`, `Welsh`, `Danish`, `German`, `Dimli (individual language)`, `Lower Sorbian`, `Dhivehi`, `Greek`, `eml`, `English`, `Esperanto`, `Spanish`, `Estonian`, `Basque`, `Extremaduran`, `Persian`, `Finnish`, `fiu-vro`, `Faroese`, `French`, `Arpitan`, `Friulian`, `Frisian`, `Irish`, `Gagauz`, `Scottish Gaelic`, `Galician`, `Guarani`, `Konkani (Goan)`, `Gujarati`, `Manx`, `Hausa`, `Hakka Chinese`, `Hebrew`, `Hindi`, `Fiji Hindi`, `Upper Sorbian`, `Haitian Creole`, `Hungarian`, `Armenian`, `Interlingua`, `Indonesian`, `Interlingue`, `Igbo`, `Ilocano`, `Ido`, `Icelandic`, `Italian`, `Japanese`, `Jamaican Patois`, `Lojban`, `Javanese`, `Georgian`, `Karakalpak`, `Kabyle`, `Kabardian`, `Kazakh`, `Khmer`, `Kannada`, `Korean`, `Komi-Permyak`, `Karachay-Balkar`, `Kölsch`, `Kurdish`, `Komi`, `Cornish`, `Kyrgyz`, `Latin`, `Ladino`, `Luxembourgish`, `Lezghian`, `Luganda`, `Limburgan`, `Ligurian`, `Lombard`, `Lingala`, `Lao`, `Northern Luri`, `Lithuanian`, `Latvian`, `Maithili`, `map-bms`, `Malagasy`, `Meadow Mari`, `Maori`, `Minangkabau`, `Macedonian`, `Malayalam`, `Mongolian`, `Marathi`, `Hill Mari`, `Maltese`, `Mirandese`, `Burmese`, `Erzya`, `Mazanderani`, `Nahuatl`, `Neapolitan`, `Low German (Low Saxon)`, `nds-nl`, `Nepali`, `Newari`, `Dutch`, `Norwegian Nynorsk`, `Norwegian`, `Narom`, `Pedi`, `Navajo`, `Occitan`, `Livvi`, `Oromo`, `Odia (Oriya)`, `Ossetian`, `Punjabi (Eastern)`, `Pangasinan`, `Kapampangan`, `Papiamento`, `Picard`, `Palatine German`, `Polish`, `Punjabi (Western)`, `Pashto`, `Portuguese`, `Quechua`, `Romansh`, `Romanian`, `roa-tara`, `Russian`, `Rusyn`, `Kinyarwanda`, `Sanskrit`, `Yakut`, `Sardinian`, `Sicilian`, `Scots`, `Sindhi`, `Northern Sami`, `Sinhala`, `Slovak`, `Slovenian`, `Shona`, `Somali`, `Albanian`, `Serbian`, `Saterland Frisian`, `Sundanese`, `Swedish`, `Swahili`, `Silesian`, `Tamil`, `Tulu`, `Telugu`, `Tetun`, `Tajik`, `Thai`, `Turkmen`, `Tagalog`, `Setswana`, `Tongan`, `Turkish`, `Tatar`, `Tuvinian`, `Udmurt`, `Uyghur`, `Ukrainian`, `Urdu`, `Uzbek`, `Venetian`, `Veps`, `Vietnamese`, `Vlaams`, `Volapük`, `Walloon`, `Waray`, `Wolof`, `Shanghainese`, `Xhosa`, `Mingrelian`, `Yiddish`, `Yoruba`, `Zeeuws`, `Chinese`, `zh-classical`, `zh-min-nan`, `zh-yue`.

## Predicted Entities

`ace`, `af`, `als`, `am`, `an`, `ang`, `ar`, `arz`, `as`, `ast`, `av`, `ay`, `az`, `azb`, `ba`, `bar`, `bat-smg`, `bcl`, `be`, `bg`, `bh`, `bn`, `bo`, `bpy`, `br`, `bxr`, `ca`, `cdo`, `ce`, `ceb`, `ckb`, `co`, `crh`, `cs`, `csb`, `cv`, `cy`, `da`, `de`, `diq`, `dsb`, `dv`, `el`, `eml`, `en`, `eo`, `es`, `et`, `eu`, `ext`, `fa`, `fi`, `fiu-vro`, `fo`, `fr`, `frp`, `fur`, `fy`, `ga`, `gag`, `gd`, `gl`, `gn`, `gom`, `gu`, `gv`, `ha`, `hak`, `he`, `hi`, `hif`, `hsb`, `ht`, `hu`, `hy`, `ia`, `id`, `ie`, `ig`, `ilo`, `io`, `is`, `it`, `ja`, `jam`, `jbo`, `jv`, `ka`, `kaa`, `kab`, `kbd`, `kk`, `km`, `kn`, `ko`, `koi`, `krc`, `ksh`, `ku`, `kv`, `kw`, `ky`, `la`, `lad`, `lb`, `lez`, `lg`, `li`, `lij`, `lmo`, `ln`, `lo`, `lrc`, `lt`, `lv`, `mai`, `map-bms`, `mg`, `mhr`, `mi`, `min`, `mk`, `ml`, `mn`, `mr`, `mrj`, `mt`, `mwl`, `my`, `myv`, `mzn`, `nah`, `nap`, `nds`, `nds-nl`, `ne`, `new`, `nl`, `nn`, `no`, `nrm`, `nso`, `nv`, `oc`, `olo`, `om`, `or`, `os`, `pa`, `pag`, `pam`, `pap`, `pcd`, `pfl`, `pl`, `pnb`, `ps`, `pt`, `qu`, `rm`, `ro`, `roa-tara`, `ru`, `rue`, `rw`, `sa`, `sah`, `sc`, `scn`, `sco`, `sd`, `se`, `si`, `sk`, `sl`, `sn`, `so`, `sq`, `sr`, `stq`, `su`, `sv`, `sw`, `szl`, `ta`, `tcy`, `te`, `tet`, `tg`, `th`, `tk`, `tl`, `tn`, `to`, `tr`, `tt`, `tyv`, `udm`, `ug`, `uk`, `ur`, `uz`, `vec`, `vep`, `vi`, `vls`, `vo`, `wa`, `war`, `wo`, `wuu`, `xh`, `xmf`, `yi`, `yo`, `zea`, `zh`, `zh-classical`, `zh-min-nan`, `zh-yue`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_tatoeba_cnn_220_xx_2.7.0_2.4_1607184539094.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_220", "xx")\
.setInputCols(["sentence"])\
.setOutputCol("language")
languagePipeline = Pipeline(stages=[documentAssembler, sentenceDetector, language_detector])
light_pipeline = LightPipeline(languagePipeline.fit(spark.createDataFrame([['']]).toDF("text")))
result = light_pipeline.fullAnnotate("Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.")
```
```scala
...
val languageDetector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_220", "xx")
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
lang_df = nlu.load('xx.classify.wiki_220').predict(text, output_level='sentence')
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
|Model Name:|ld_wiki_tatoeba_cnn_220|
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
|      sv| 1000|    999|             0.999|
|      fr| 1000|    999|             0.999|
|      fi| 1000|    998|             0.998|
|      it| 1000|    997|             0.997|
|      pt| 1000|    995|             0.995|
|      el| 1000|    994|             0.994|
|      de| 1000|    993|             0.993|
|      en| 1000|    990|              0.99|
|      nl| 1000|    987|             0.987|
|      hu|  880|    866|0.9840909090909091|
|      da| 1000|    980|              0.98|
|      es| 1000|    976|             0.976|
|      ro|  784|    765|0.9757653061224489|
|      et|  928|    905|0.9752155172413793|
|      lt| 1000|    975|             0.975|
|      cs| 1000|    973|             0.973|
|      pl|  914|    889|0.9726477024070022|
|      sk| 1000|    941|             0.941|
|      bg| 1000|    939|             0.939|
|      lv|  916|    857|0.9355895196506551|
|      sl|  914|    789|0.8632385120350109|
+--------+-----+-------+------------------+

+-------+-------------------+
|summary|          precision|
+-------+-------------------+
|  count|                 21|
|   mean| 0.9734546412641623|
| stddev|0.03176749551086062|
|    min| 0.8632385120350109|
|    max|              0.999|
+-------+-------------------+

```