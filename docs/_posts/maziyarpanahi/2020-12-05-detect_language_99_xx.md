---
layout: model
title: Language Detection & Identification Pipeline - 99 Languages
author: John Snow Labs
name: detect_language_99
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
`Afrikaans`, `Arabic`, `Algerian Arabic`, `Assamese`, `Kotava`, `Azerbaijani`, `Belarusian`, `Bengali`, `Berber`, `Breton`, `Bulgarian`, `Catalan`, `Chavacano`, `Cebuano`, `Czech`, `Chuvash`, `Mandarin Chinese`, `Cornish`, `Danish`, `German`, `Central Dusun`, `Modern Greek (1453-)`, `English`, `Esperanto`, `Estonian`, `Basque`, `Finnish`, `French`, `Guadeloupean Creole French`, `Irish`, `Galician`, `Gronings`, `Guarani`, `Hebrew`, `Hindi`, `Croatian`, `Hungarian`, `Armenian`, `Ido`, `Interlingue`, `Ilocano`, `Interlingua`, `Indonesian`, `Icelandic`, `Italian`, `Lojban`, `Japanese`, `Kabyle`, `Georgian`, `Kazakh`, `Khasi`, `Khmer`, `Korean`, `Coastal Kadazan`, `Latin`, `Lingua Franca Nova`, `Lithuanian`, `Latvian`, `Literary Chinese`, `Marathi`, `Meadow Mari`, `Macedonian`, `Low German (Low Saxon)`, `Dutch`, `Norwegian Nynorsk`, `Norwegian Bokmål`, `Occitan`, `Ottoman Turkish`, `Kapampangan`, `Picard`, `Persian`, `Polish`, `Portuguese`, `Romanian`, `Kirundi`, `Russian`, `Slovak`, `Spanish`, `Albanian`, `Serbian`, `Swedish`, `Swabian`, `Tatar`, `Tagalog`, `Thai`, `Klingon`, `Toki Pona`, `Turkmen`, `Turkish`, `Uyghur`, `Ukrainian`, `Urdu`, `Vietnamese`, `Volapük`, `Waray`, `Shanghainese`, `Yiddish`, `Cantonese`, `Malay`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/LANGUAGE_DETECTOR/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_99_xx_2.7.0_2.4_1607185604600.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("detect_language_99", lang = "xx")

pipeline.annotate("French author who helped pioneer the science-fiction genre.")
```

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("detect_language_99", lang = "xx")

pipeline.annotate("French author who helped pioneer the science-fiction genre.")
```

{:.nlu-block}
```python
import nlu

text = ["French author who helped pioneer the science-fiction genre."]
lang_df = nlu.load("xx.classify.lang.99").predict(text)
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
|Model Name:|detect_language_99|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Language:|xx|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- LanguageDetectorDL