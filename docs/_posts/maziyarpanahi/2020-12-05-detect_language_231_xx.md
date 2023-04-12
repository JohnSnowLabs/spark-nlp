---
layout: model
title: Language Detection & Identification Pipeline - 231 Languages
author: John Snow Labs
name: detect_language_231
date: 2020-12-05
task: [Pipeline Public, Language Detection, Sentence Detection]
language: xx
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [language_detection, open_source, pipeline, xx]
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Language detection and identification is the task of automatically detecting the language(s) present in a document based on the content of the document. ``LanguageDetectorDL`` is an annotator that detects the language of documents or sentences depending on the ``inputCols``. In addition, ``LanguageDetetorDL`` can accurately detect language from documents with mixed languages by coalescing sentences and select the best candidate.

We have designed and developed Deep Learning models using CNN architectures in TensorFlow/Keras. The models are trained on large datasets such as Wikipedia and Tatoeba with high accuracy evaluated on the Europarl dataset. The output is a language code in Wiki Code style: [https://en.wikipedia.org/wiki/List_of_Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias).

This pipeline can detect the following languages:

## Predicted Entities
`Achinese`, `Afrikaans`, `Tosk Albanian`, `Amharic`, `Aragonese`, `Old English`, `Arabic`, `Egyptian Arabic`, `Assamese`, `Asturian`, `Avaric`, `Aymara`, `Azerbaijani`, `South Azerbaijani`, `Bashkir`, `Bavarian`, `bat-smg`, `Central Bikol`, `Belarusian`, `Bulgarian`, `bh`, `Banjar`, `Bengali`, `Tibetan`, `Bishnupriya`, `Breton`, `Bosnian`, `Russia Buriat`, `Catalan`, `cbk-zam`, `Min Dong Chinese`, `Chechen`, `Cebuano`, `Central Kurdish (Soranî)`, `Corsican`, `Crimean Tatar`, `Czech`, `Kashubian`, `Chuvash`, `Welsh`, `Danish`, `German`, `Dimli (individual language)`, `Lower Sorbian`, `dty`, `Dhivehi`, `Greek`, `eml`, `English`, `Esperanto`, `Spanish`, `Estonian`, `Extremaduran`, `Persian`, `Finnish`, `fiu-vro`, `Faroese`, `French`, `Arpitan`, `Friulian`, `Frisian`, `Irish`, `Gagauz`, `Scottish Gaelic`, `Galician`, `Gilaki`, `Guarani`, `Konkani (Goan)`, `Gujarati`, `Manx`, `Hausa`, `Hakka Chinese`, `Hebrew`, `Hindi`, `Fiji Hindi`, `Croatian`, `Upper Sorbian`, `Haitian Creole`, `Hungarian`, `Armenian`, `Interlingua`, `Indonesian`, `Interlingue`, `Igbo`, `Ilocano`, `Ido`, `Icelandic`, `Italian`, `Japanese`, `Jamaican Patois`, `Lojban`, `Javanese`, `Georgian`, `Karakalpak`, `Kabyle`, `Kabardian`, `Kazakh`, `Khmer`, `Kannada`, `Korean`, `Komi-Permyak`, `Karachay-Balkar`, `Kölsch`, `Kurdish`, `Komi`, `Cornish`, `Kyrgyz`, `Latin`, `Ladino`, `Luxembourgish`, `Lezghian`, `Luganda`, `Limburgan`, `Ligurian`, `Lombard`, `Lingala`, `Lao`, `Northern Luri`, `Lithuanian`, `Latgalian`, `Latvian`, `Maithili`, `map-bms`, `Moksha`, `Malagasy`, `Meadow Mari`, `Maori`, `Minangkabau`, `Macedonian`, `Malayalam`, `Mongolian`, `Marathi`, `Hill Mari`, `Malay`, `Maltese`, `Mirandese`, `Burmese`, `Erzya`, `Mazanderani`, `Nahuatl`, `Neapolitan`, `Low German (Low Saxon)`, `nds-nl`, `Nepali`, `Newari`, `Dutch`, `Norwegian Nynorsk`, `Norwegian`, `Narom`, `Pedi`, `Navajo`, `Occitan`, `Livvi`, `Oromo`, `Odia (Oriya)`, `Ossetian`, `Punjabi (Eastern)`, `Pangasinan`, `Kapampangan`, `Papiamento`, `Picard`, `Pennsylvania German`, `Palatine German`, `Polish`, `Punjabi (Western)`, `Pashto`, `Portuguese`, `Quechua`, `Romansh`, `Romanian`, `roa-rup`, `roa-tara`, `Russian`, `Rusyn`, `Kinyarwanda`, `Sanskrit`, `Yakut`, `Sardinian`, `Sicilian`, `Sindhi`, `Northern Sami`, `Serbo-Croatian`, `Sinhala`, `Slovak`, `Slovenian`, `Shona`, `Somali`, `Albanian`, `Serbian`, `Sranan Tongo`, `Saterland Frisian`, `Sundanese`, `Swedish`, `Swahili`, `Silesian`, `Tamil`, `Tulu`, `Telugu`, `Tetun`, `Tajik`, `Thai`, `Turkmen`, `Tagalog`, `Setswana`, `Tongan`, `Turkish`, `Tatar`, `Tuvinian`, `Udmurt`, `Uyghur`, `Ukrainian`, `Urdu`, `Uzbek`, `Venetian`, `Veps`, `Vietnamese`, `Vlaams`, `Volapük`, `Walloon`, `Waray`, `Wolof`, `Shanghainese`, `Xhosa`, `Mingrelian`, `Yiddish`, `Yoruba`, `Zeeuws`, `Chinese`, `zh-classical`, `zh-min-nan`, `zh-yue`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/LANGUAGE_DETECTOR/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_231_xx_2.7.0_2.4_1607185843755.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/detect_language_231_xx_2.7.0_2.4_1607185843755.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("detect_language_231", lang = "xx")

pipeline.annotate("French author who helped pioneer the science-fiction genre.")
```

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("detect_language_231", lang = "xx")

pipeline.annotate("French author who helped pioneer the science-fiction genre.")
```

{:.nlu-block}
```python
import nlu

text = ["French author who helped pioneer the science-fiction genre."]
lang_df = nlu.load("xx.classify.lang.231").predict(text)
lang_df
```

</div>

## Results

```bash
{'document': ['French author who helped pioneer the science-fiction genre.'],
'language': ['en']}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|detect_language_231|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Language:|xx|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- LanguageDetectorDL