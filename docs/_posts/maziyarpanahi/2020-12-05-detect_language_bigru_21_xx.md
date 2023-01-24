---
layout: model
title: Language Detection & Identification Pipeline - 21 Languages (BiGRU)
author: John Snow Labs
name: detect_language_bigru_21
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

We have designed and developed Deep Learning models using BiGRU architectures (mentioned in the model's name) in TensorFlow/Keras. The models are trained on large datasets such as Wikipedia and Tatoeba with high accuracy evaluated on the Europarl dataset. The output is a language code in Wiki Code style: [https://en.wikipedia.org/wiki/List_of_Wikipedias](https://en.wikipedia.org/wiki/List_of_Wikipedias).

This pipeline can detect the following languages:

## Predicted Entities
`Bulgarian`, `Czech`, `Danish`, `German`, `Greek`, `English`, `Estonian`, `Finnish`, `French`, `Hungarian`, `Italian`, `Lithuanian`, `Latvian`, `Dutch`, `Polish`, `Portuguese`, `Romanian`, `Slovak`, `Slovenian`, `Spanish`, `Swedish`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/LANGUAGE_DETECTOR/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/detect_language_bigru_21_xx_2.7.0_2.4_1607186103596.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/detect_language_bigru_21_xx_2.7.0_2.4_1607186103596.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("detect_language_bigru_21", lang = "xx")

pipeline.annotate("French author who helped pioneer the science-fiction genre.")
```

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("detect_language_bigru_21", lang = "xx")

pipeline.annotate("French author who helped pioneer the science-fiction genre.")
```

{:.nlu-block}
```python
import nlu

text = ["French author who helped pioneer the science-fiction genre."]
lang_df = nlu.load("xx.classify.lang.bigru").predict(text)
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
|Model Name:|detect_language_bigru_21|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Language:|xx|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- LanguageDetectorDL