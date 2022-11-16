---
layout: model
title: Fast and Accurate Language Identification - 375 Languages (CNN)
author: John Snow Labs
name: ld_wiki_tatoeba_cnn_375
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

`Abkhaz`, `Iraqi Arabic`, `Adyghe`, `Afrikaans`, `Gulf Arabic`, `Afrihili`, `Assyrian Neo-Aramaic`, `Ainu`, `Aklanon`, `Gheg Albanian`, `Amharic`, `Aragonese`, `Old English`, `Uab Meto`, `North Levantine Arabic`, `Arabic`, `Algerian Arabic`, `Moroccan Arabic`, `Egyptian Arabic`, `Assamese`, `Asturian`, `Kotava`, `Awadhi`, `Aymara`, `Azerbaijani`, `Bashkir`, `Baluchi`, `Balinese`, `Bavarian`, `Central Bikol`, `Belarusian`, `Berber`, `Bulgarian`, `Bhojpuri`, `Bislama`, `Banjar`, `Bambara`, `Bengali`, `Tibetan`, `Breton`, `Bodo`, `Bosnian`, `Buryat`, `Baybayanon`, `Brithenig`, `Catalan`, `Cayuga`, `Chavacano`, `Chechen`, `Cebuano`, `Chamorro`, `Chagatai`, `Chinook Jargon`, `Choctaw`, `Cherokee`, `Jin Chinese`, `Chukchi`, `Central Mnong`, `Corsican`, `Chinese Pidgin English`, `Crimean Tatar`, `Seychellois Creole`, `Czech`, `Kashubian`, `Chuvash`, `Welsh`, `CycL`, `Cuyonon`, `Danish`, `German`, `Dungan`, `Drents`, `Lower Sorbian`, `Central Dusun`, `Dhivehi`, `Dutton World Speedwords`, `Ewe`, `Emilian`, `Greek`, `Erromintxela`, `English`, `Middle English`, `Esperanto`, `Spanish`, `Estonian`, `Basque`, `Evenki`, `Extremaduran`, `Persian`, `Finnish`, `Fijian`, `Kven Finnish`, `Faroese`, `French`, `Middle French`, `Old French`, `North Frisian`, `Pulaar`, `Friulian`, `Nigerian Fulfulde`, `Frisian`, `Irish`, `Ga`, `Gagauz`, `Gan Chinese`, `Garhwali`, `Guadeloupean Creole French`, `Scottish Gaelic`, `Gilbertese`, `Galician`, `Guarani`, `Konkani (Goan)`, `Gronings`, `Gothic`, `Ancient Greek`, `Swiss German`, `Gujarati`, `Manx`, `Hausa`, `Hakka Chinese`, `Hawaiian`, `Ancient Hebrew`, `Hebrew`, `Hindi`, `Fiji Hindi`, `Hiligaynon`, `Hmong Njua (Green)`, `Ho`, `Croatian`, `Hunsrik`, `Upper Sorbian`, `Xiang Chinese`, `Haitian Creole`, `Hungarian`, `Armenian`, `Interlingua`, `Iban`, `Indonesian`, `Interlingue`, `Igbo`, `Nuosu`, `Inuktitut`, `Ilocano`, `Ido`, `Icelandic`, `Italian`, `Ingrian`, `Japanese`, `Jamaican Patois`, `Lojban`, `Juhuri (Judeo-Tat)`, `Jewish Palestinian Aramaic`, `Javanese`, `Georgian`, `Karakalpak`, `Kabyle`, `Kamba`, `Kekchi (Q'eqchi')`, `Khasi`, `Khakas`, `Kazakh`, `Greenlandic`, `Khmer`, `Kannada`, `Korean`, `Komi-Permyak`, `Komi-Zyrian`, `Karachay-Balkar`, `Karelian`, `Kashmiri`, `Kölsch`, `Kurdish`, `Kumyk`, `Cornish`, `Keningau Murut`, `Kyrgyz`, `Coastal Kadazan`, `Latin`, `Southern Subanen`, `Ladino`, `Luxembourgish`, `Láadan`, `Lingua Franca Nova`, `Luganda`, `Ligurian`, `Livonian`, `Lakota`, `Ladin`, `Lombard`, `Lingala`, `Lao`, `Louisiana Creole`, `Lithuanian`, `Latgalian`, `Latvian`, `Latvian`, `Literary Chinese`, `Laz`, `Madurese`, `Maithili`, `North Moluccan Malay`, `Moksha`, `Morisyen`, `Malagasy`, `Mambae`, `Marshallese`, `Meadow Mari`, `Maori`, `Mi'kmaq`, `Minangkabau`, `Macedonian`, `Malayalam`, `Mongolian`, `Manchu`, `Mon`, `Mohawk`, `Marathi`, `Hill Mari`, `Malay`, `Maltese`, `Tagal Murut`, `Mirandese`, `Hmong Daw (White)`, `Burmese`, `Erzya`, `Nauruan`, `Nahuatl`, `Norwegian Bokmål`, `Central Huasteca Nahuatl`, `Low German (Low Saxon)`, `Nepali`, `Newari`, `Ngeq`, `Guerrero Nahuatl`, `Niuean`, `Dutch`, `Orizaba Nahuatl`, `Norwegian Nynorsk`, `Norwegian`, `Nogai`, `Old Norse`, `Novial`, `Nepali`, `Naga (Tangshang)`, `Navajo`, `Chinyanja`, `Nyungar`, `Old Aramaic`, `Occitan`, `Ojibwe`, `Odia (Oriya)`, `Old East Slavic`, `Ossetian`, `Old Spanish`, `Old Saxon`, `Ottoman Turkish`, `Old Turkish`, `Punjabi (Eastern)`, `Pangasinan`, `Kapampangan`, `Papiamento`, `Palauan`, `Picard`, `Pennsylvania German`, `Palatine German`, `Phoenician`, `Pali`, `Polish`, `Piedmontese`, `Punjabi (Western)`, `Pipil`, `Old Prussian`, `Pashto`, `Portuguese`, `Quechua`, `K'iche'`, `Quenya`, `Rapa Nui`, `Rendille`, `Tarifit`, `Romansh`, `Kirundi`, `Romanian`, `Romani`, `Russian`, `Rusyn`, `Kinyarwanda`, `Okinawan`, `Sanskrit`, `Yakut`, `Sardinian`, `Sicilian`, `Scots`, `Sindhi`, `Northern Sami`, `Sango`, `Samogitian`, `Shuswap`, `Tachawit`, `Sinhala`, `Sindarin`, `Slovak`, `Slovenian`, `Samoan`, `Southern Sami`, `Shona`, `Somali`, `Albanian`, `Serbian`, `Swazi`, `Southern Sotho`, `Saterland Frisian`, `Sundanese`, `Sumerian`, `Swedish`, `Swahili`, `Swabian`, `Swahili`, `Syriac`, `Tamil`, `Telugu`, `Tetun`, `Tajik`, `Thai`, `Tahaggart Tamahaq`, `Tigrinya`, `Tigre`, `Turkmen`, `Tokelauan`, `Tagalog`, `Klingon`, `Talysh`, `Jewish Babylonian Aramaic`, `Temuan`, `Setswana`, `Tongan`, `Tonga (Zambezi)`, `Toki Pona`, `Tok Pisin`, `Old Tupi`, `Turkish`, `Tsonga`, `Tatar`, `Isan`, `Tuvaluan`, `Tahitian`, `Tuvinian`, `Talossan`, `Udmurt`, `Uyghur`, `Ukrainian`, `Umbundu`, `Urdu`, `Urhobo`, `Uzbek`, `Venetian`, `Veps`, `Vietnamese`, `Volapük`, `Võro`, `Walloon`, `Waray`, `Wolof`, `Shanghainese`, `Kalmyk`, `Xhosa`, `Mingrelian`, `Yiddish`, `Yoruba`, `Cantonese`, `Chinese`, `Malay (Vernacular)`, `Malay`, `Zulu`, `Zaza`.

## Predicted Entities

`ab`, `acm`, `ady`, `af`, `afb`, `afh`, `aii`, `ain`, `akl`, `aln`, `am`, `an`, `ang`, `aoz`, `apc`, `ar`, `arq`, `ary`, `arz`, `as`, `ast`, `avk`, `awa`, `ay`, `az`, `ba`, `bal`, `ban`, `bar`, `bcl`, `be`, `ber`, `bg`, `bho`, `bi`, `bjn`, `bm`, `bn`, `bo`, `br`, `brx`, `bs`, `bua`, `bvy`, `bzt`, `ca`, `cay`, `cbk`, `ce`, `ceb`, `ch`, `chg`, `chn`, `cho`, `chr`, `cjy`, `ckt`, `cmo`, `co`, `cpi`, `crh`, `crs`, `cs`, `csb`, `cv`, `cy`, `cycl`, `cyo`, `da`, `de`, `dng`, `drt`, `dsb`, `dtp`, `dv`, `dws`, `ee`, `egl`, `el`, `emx`, `en`, `enm`, `eo`, `es`, `et`, `eu`, `evn`, `ext`, `fa`, `fi`, `fj`, `fkv`, `fo`, `fr`, `frm`, `fro`, `frr`, `fuc`, `fur`, `fuv`, `fy`, `ga`, `gaa`, `gag`, `gan`, `gbm`, `gcf`, `gd`, `gil`, `gl`, `gn`, `gom`, `gos`, `got`, `grc`, `gsw`, `gu`, `gv`, `ha`, `hak`, `haw`, `hbo`, `he`, `hi`, `hif`, `hil`, `hnj`, `hoc`, `hr`, `hrx`, `hsb`, `hsn`, `ht`, `hu`, `hy`, `ia`, `iba`, `id`, `ie`, `ig`, `ii`, `ike`, `ilo`, `io`, `is`, `it`, `izh`, `ja`, `jam`, `jbo`, `jdt`, `jpa`, `jv`, `ka`, `kaa`, `kab`, `kam`, `kek`, `kha`, `kjh`, `kk`, `kl`, `km`, `kn`, `ko`, `koi`, `kpv`, `krc`, `krl`, `ks`, `ksh`, `ku`, `kum`, `kw`, `kxi`, `ky`, `kzj`, `la`, `laa`, `lad`, `lb`, `ldn`, `lfn`, `lg`, `lij`, `liv`, `lkt`, `lld`, `lmo`, `ln`, `lo`, `lou`, `lt`, `ltg`, `lv`, `lvs`, `lzh`, `lzz`, `mad`, `mai`, `max`, `mdf`, `mfe`, `mg`, `mgm`, `mh`, `mhr`, `mi`, `mic`, `min`, `mk`, `ml`, `mn`, `mnc`, `mnw`, `moh`, `mr`, `mrj`, `ms`, `mt`, `mvv`, `mwl`, `mww`, `my`, `myv`, `na`, `nah`, `nb`, `nch`, `nds`, `ne`, `new`, `ngt`, `ngu`, `niu`, `nl`, `nlv`, `nn`, `no`, `nog`, `non`, `nov`, `npi`, `nst`, `nv`, `ny`, `nys`, `oar`, `oc`, `oj`, `or`, `orv`, `os`, `osp`, `osx`, `ota`, `otk`, `pa`, `pag`, `pam`, `pap`, `pau`, `pcd`, `pdc`, `pfl`, `phn`, `pi`, `pl`, `pms`, `pnb`, `ppl`, `prg`, `ps`, `pt`, `qu`, `quc`, `qya`, `rap`, `rel`, `rif`, `rm`, `rn`, `ro`, `rom`, `ru`, `rue`, `rw`, `ryu`, `sa`, `sah`, `sc`, `scn`, `sco`, `sd`, `se`, `sg`, `sgs`, `shs`, `shy`, `si`, `sjn`, `sk`, `sl`, `sm`, `sma`, `sn`, `so`, `sq`, `sr`, `ss`, `st`, `stq`, `su`, `sux`, `sv`, `sw`, `swg`, `swh`, `syc`, `ta`, `te`, `tet`, `tg`, `th`, `thv`, `ti`, `tig`, `tk`, `tkl`, `tl`, `tlh`, `tly`, `tmr`, `tmw`, `tn`, `to`, `toi`, `toki`, `tpi`, `tpw`, `tr`, `ts`, `tt`, `tts`, `tvl`, `ty`, `tyv`, `tzl`, `udm`, `ug`, `uk`, `umb`, `ur`, `urh`, `uz`, `vec`, `vep`, `vi`, `vo`, `vro`, `wa`, `war`, `wo`, `wuu`, `xal`, `xh`, `xmf`, `yi`, `yo`, `yue`, `zh`, `zlm`, `zsm`, `zu`, `zza`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ld_wiki_tatoeba_cnn_375_xx_2.7.0_2.4_1607184873730.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
language_detector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_375", "xx")\
.setInputCols(["sentence"])\
.setOutputCol("language")
languagePipeline = Pipeline(stages=[documentAssembler, sentenceDetector, language_detector])
light_pipeline = LightPipeline(languagePipeline.fit(spark.createDataFrame([['']]).toDF("text")))
result = light_pipeline.fullAnnotate("Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala.")
```
```scala
...
val languageDetector = LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_375", "xx")
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
lang_df = nlu.load('xx.classify.wiki_375').predict(text, output_level='sentence')
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
|Model Name:|ld_wiki_tatoeba_cnn_375|
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
|      de| 1000|    999|             0.999|
|      fi| 1000|    999|             0.999|
|      nl| 1000|    998|             0.998|
|      el| 1000|    997|             0.997|
|      en| 1000|    995|             0.995|
|      es| 1000|    994|             0.994|
|      it| 1000|    993|             0.993|
|      sv| 1000|    991|             0.991|
|      da| 1000|    987|             0.987|
|      pl|  914|    901|0.9857768052516411|
|      hu|  880|    866|0.9840909090909091|
|      pt| 1000|    980|              0.98|
|      et|  928|    907|0.9773706896551724|
|      ro|  784|    766|0.9770408163265306|
|      lt| 1000|    976|             0.976|
|      bg| 1000|    965|             0.965|
|      cs| 1000|    945|             0.945|
|      sk| 1000|    944|             0.944|
|      lv|  916|    843|0.9203056768558951|
|      sl|  914|    810|0.8862144420131292|
+--------+-----+-------+------------------+

+-------+--------------------+
|summary|           precision|
+-------+--------------------+
|  count|                  21|
|   mean|  0.9758952066282511|
| stddev|0.029434744995013935|
|    min|  0.8862144420131292|
|    max|                 1.0|
+-------+--------------------+
```