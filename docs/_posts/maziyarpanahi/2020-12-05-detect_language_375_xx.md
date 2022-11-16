---
layout: model
title: Language Detection & Identification Pipeline - 375 Languages
author: John Snow Labs
name: detect_language_375
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
`Abkhaz`, `Iraqi Arabic`, `Adyghe`, `Afrikaans`, `Gulf Arabic`, `Afrihili`, `Assyrian Neo-Aramaic`, `Ainu`, `Aklanon`, `Gheg Albanian`, `Amharic`, `Aragonese`, `Old English`, `Uab Meto`, `North Levantine Arabic`, `Arabic`, `Algerian Arabic`, `Moroccan Arabic`, `Egyptian Arabic`, `Assamese`, `Asturian`, `Kotava`, `Awadhi`, `Aymara`, `Azerbaijani`, `Bashkir`, `Baluchi`, `Balinese`, `Bavarian`, `Central Bikol`, `Belarusian`, `Berber`, `Bulgarian`, `Bhojpuri`, `Bislama`, `Banjar`, `Bambara`, `Bengali`, `Tibetan`, `Breton`, `Bodo`, `Bosnian`, `Buryat`, `Baybayanon`, `Brithenig`, `Catalan`, `Cayuga`, `Chavacano`, `Chechen`, `Cebuano`, `Chamorro`, `Chagatai`, `Chinook Jargon`, `Choctaw`, `Cherokee`, `Jin Chinese`, `Chukchi`, `Central Mnong`, `Corsican`, `Chinese Pidgin English`, `Crimean Tatar`, `Seychellois Creole`, `Czech`, `Kashubian`, `Chuvash`, `Welsh`, `CycL`, `Cuyonon`, `Danish`, `German`, `Dungan`, `Drents`, `Lower Sorbian`, `Central Dusun`, `Dhivehi`, `Dutton World Speedwords`, `Ewe`, `Emilian`, `Greek`, `Erromintxela`, `English`, `Middle English`, `Esperanto`, `Spanish`, `Estonian`, `Basque`, `Evenki`, `Extremaduran`, `Persian`, `Finnish`, `Fijian`, `Kven Finnish`, `Faroese`, `French`, `Middle French`, `Old French`, `North Frisian`, `Pulaar`, `Friulian`, `Nigerian Fulfulde`, `Frisian`, `Irish`, `Ga`, `Gagauz`, `Gan Chinese`, `Garhwali`, `Guadeloupean Creole French`, `Scottish Gaelic`, `Gilbertese`, `Galician`, `Guarani`, `Konkani (Goan)`, `Gronings`, `Gothic`, `Ancient Greek`, `Swiss German`, `Gujarati`, `Manx`, `Hausa`, `Hakka Chinese`, `Hawaiian`, `Ancient Hebrew`, `Hebrew`, `Hindi`, `Fiji Hindi`, `Hiligaynon`, `Hmong Njua (Green)`, `Ho`, `Croatian`, `Hunsrik`, `Upper Sorbian`, `Xiang Chinese`, `Haitian Creole`, `Hungarian`, `Armenian`, `Interlingua`, `Iban`, `Indonesian`, `Interlingue`, `Igbo`, `Nuosu`, `Inuktitut`, `Ilocano`, `Ido`, `Icelandic`, `Italian`, `Ingrian`, `Japanese`, `Jamaican Patois`, `Lojban`, `Juhuri (Judeo-Tat)`, `Jewish Palestinian Aramaic`, `Javanese`, `Georgian`, `Karakalpak`, `Kabyle`, `Kamba`, `Kekchi (Q'eqchi')`, `Khasi`, `Khakas`, `Kazakh`, `Greenlandic`, `Khmer`, `Kannada`, `Korean`, `Komi-Permyak`, `Komi-Zyrian`, `Karachay-Balkar`, `Karelian`, `Kashmiri`, `Kölsch`, `Kurdish`, `Kumyk`, `Cornish`, `Keningau Murut`, `Kyrgyz`, `Coastal Kadazan`, `Latin`, `Southern Subanen`, `Ladino`, `Luxembourgish`, `Láadan`, `Lingua Franca Nova`, `Luganda`, `Ligurian`, `Livonian`, `Lakota`, `Ladin`, `Lombard`, `Lingala`, `Lao`, `Louisiana Creole`, `Lithuanian`, `Latgalian`, `Latvian`, `Latvian`, `Literary Chinese`, `Laz`, `Madurese`, `Maithili`, `North Moluccan Malay`, `Moksha`, `Morisyen`, `Malagasy`, `Mambae`, `Marshallese`, `Meadow Mari`, `Maori`, `Mi'kmaq`, `Minangkabau`, `Macedonian`, `Malayalam`, `Mongolian`, `Manchu`, `Mon`, `Mohawk`, `Marathi`, `Hill Mari`, `Malay`, `Maltese`, `Tagal Murut`, `Mirandese`, `Hmong Daw (White)`, `Burmese`, `Erzya`, `Nauruan`, `Nahuatl`, `Norwegian Bokmål`, `Central Huasteca Nahuatl`, `Low German (Low Saxon)`, `Nepali`, `Newari`, `Ngeq`, `Guerrero Nahuatl`, `Niuean`, `Dutch`, `Orizaba Nahuatl`, `Norwegian Nynorsk`, `Norwegian`, `Nogai`, `Old Norse`, `Novial`, `Nepali`, `Naga (Tangshang)`, `Navajo`, `Chinyanja`, `Nyungar`, `Old Aramaic`, `Occitan`, `Ojibwe`, `Odia (Oriya)`, `Old East Slavic`, `Ossetian`, `Old Spanish`, `Old Saxon`, `Ottoman Turkish`, `Old Turkish`, `Punjabi (Eastern)`, `Pangasinan`, `Kapampangan`, `Papiamento`, `Palauan`, `Picard`, `Pennsylvania German`, `Palatine German`, `Phoenician`, `Pali`, `Polish`, `Piedmontese`, `Punjabi (Western)`, `Pipil`, `Old Prussian`, `Pashto`, `Portuguese`, `Quechua`, `K'iche'`, `Quenya`, `Rapa Nui`, `Rendille`, `Tarifit`, `Romansh`, `Kirundi`, `Romanian`, `Romani`, `Russian`, `Rusyn`, `Kinyarwanda`, `Okinawan`, `Sanskrit`, `Yakut`, `Sardinian`, `Sicilian`, `Scots`, `Sindhi`, `Northern Sami`, `Sango`, `Samogitian`, `Shuswap`, `Tachawit`, `Sinhala`, `Sindarin`, `Slovak`, `Slovenian`, `Samoan`, `Southern Sami`, `Shona`, `Somali`, `Albanian`, `Serbian`, `Swazi`, `Southern Sotho`, `Saterland Frisian`, `Sundanese`, `Sumerian`, `Swedish`, `Swahili`, `Swabian`, `Swahili`, `Syriac`, `Tamil`, `Telugu`, `Tetun`, `Tajik`, `Thai`, `Tahaggart Tamahaq`, `Tigrinya`, `Tigre`, `Turkmen`, `Tokelauan`, `Tagalog`, `Klingon`, `Talysh`, `Jewish Babylonian Aramaic`, `Temuan`, `Setswana`, `Tongan`, `Tonga (Zambezi)`, `Toki Pona`, `Tok Pisin`, `Old Tupi`, `Turkish`, `Tsonga`, `Tatar`, `Isan`, `Tuvaluan`, `Tahitian`, `Tuvinian`, `Talossan`, `Udmurt`, `Uyghur`, `Ukrainian`, `Umbundu`, `Urdu`, `Urhobo`, `Uzbek`, `Venetian`, `Veps`, `Vietnamese`, `Volapük`, `Võro`, `Walloon`, `Waray`, `Wolof`, `Shanghainese`, `Kalmyk`, `Xhosa`, `Mingrelian`, `Yiddish`, `Yoruba`, `Cantonese`, `Chinese`, `Malay (Vernacular)`, `Malay`, `Zulu`, `Zaza`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/LANGUAGE_DETECTOR/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_375_xx_2.7.0_2.4_1607185980306.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("detect_language_375", lang = "xx")

pipeline.annotate("French author who helped pioneer the science-fiction genre.")
```

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("detect_language_375", lang = "xx")

pipeline.annotate("French author who helped pioneer the science-fiction genre.")
```

{:.nlu-block}
```python
import nlu

text = ["French author who helped pioneer the science-fiction genre."]
lang_df = nlu.load("lang").predict(text)
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
|Model Name:|detect_language_375|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Language:|xx|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- LanguageDetectorDL