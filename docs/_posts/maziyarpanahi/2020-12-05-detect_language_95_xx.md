---
layout: model
title: Language Detection & Identification Pipeline - 95 Languages
author: John Snow Labs
name: detect_language_95
date: 2020-12-05
tags: [language_detection, open_source, pipeline, xx]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Language detection and identification is the task of automatically detecting the language(s) present in a document based on the content of the document. LanguageDetectorDL is an annotator that detects the language of documents or sentences depending on the inputCols. In addition, LanguageDetetorDL can accurately detect language from documents with mixed languages by coalescing sentences and select the best candidate.


We have designed and developed Deep Learning models by using CNNs and BiGRU architectures (mentioned in the model's name) in TensorFlow/Keras. The models are trained on large datasets such as Wikipedia and Tatoeba with high accuracy evaluated on the Europarl dataset. The output is a language code in Wiki Code style: [https://en.wikipedia.org/wiki/List_of_Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias)

This model can detect the following languages:

\[`Afrikaans`, `Amharic`, `Aragonese`, `Arabic`, `Assamese`, `Azerbaijani`, `Belarusian`, `Bulgarian`, `Bengali`, `Breton`, `Bosnian`, `Catalan`, `Czech`, `Welsh`, `Danish`, `German`, `Greek`, `English`, `Esperanto`, `Spanish`, `Estonian`, `Basque`, `Persian`, `Finnish`, `Faroese`, `French`, `Irish`, `Galician`, `Gujarati`, `Hebrew`, `Hindi`, `Croatian`, `Haitian Creole`, `Hungarian`, `Armenian`, `Interlingua`, `Indonesian`, `Icelandic`, `Italian`, `Japanese`, `Javanese`, `Georgian`, `Kazakh`, `Khmer`, `Kannada`, `Korean`, `Kurdish`, `Kyrgyz`, `Latin`, `Luxembourgish`, `Lao`, `Lithuanian`, `Latvian`, `Malagasy`, `Macedonian`, `Malayalam`, `Mongolian`, `Marathi`, `Malay`, `Maltese`, `Nepali`, `Dutch`, `Norwegian Nynorsk`, `Norwegian`, `Occitan`, `Odia (Oriya)`, `Punjabi (Eastern)`, `Polish`, `Pashto`, `Portuguese`, `Quechua`, `Romanian`, `Russian`, `Northern Sami`, `Sinhala`, `Slovak`, `Slovenian`, `Albanian`, `Serbian`, `Swedish`, `Swahili`, `Tamil`, `Telugu`, `Thai`, `Tagalog`, `Turkish`, `Tatar`, `Uyghur`, `Ukrainian`, `Urdu`, `Vietnamese`, `Volapük`, `Walloon`, `Xhosa`, `Chinese`]

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_95_xx_2.7.0_2.4_1607185479059.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline('detect_language_95')

pipeline.annotate("French author who helped pioneer the science-fiction genre.")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("detect_language_95")

pipeline.annotate("French author who helped pioneer the science-fiction genre.")

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
|Model Name:|detect_language_95|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Language:|xx|

## Included Models

\- DocumentAssembler
\- SentenceDetectorDLModel
\- LanguageDetectorDL