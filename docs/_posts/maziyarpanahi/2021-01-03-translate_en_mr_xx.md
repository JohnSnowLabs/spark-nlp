---
layout: model
title: Translate English to Marathi Pipeline
author: John Snow Labs
name: translate_en_mr
date: 2021-01-03
task: [Translation, Pipeline Public]
language: xx
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [open_source, seq2seq, translation, pipeline, en, mr, xx]
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies. It is mainly being developed by the Microsoft Translator team. Many academic (most notably the University of Edinburgh and in the past the Adam Mickiewicz University in Poznań) and commercial contributors help with its development.

It is currently the engine behind the Microsoft Translator Neural Machine Translation services and being deployed by many companies, organizations and research projects (see below for an incomplete list).

Note that this is a very computationally expensive module especially on larger sequence. The use of an accelerator such as GPU is recommended.

- source languages: `en`

- target languages: `mr`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/INDIAN_TRANSLATION_MARATHI/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/TRANSLATION_PIPELINES_MODELS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/translate_en_mr_xx_2.7.0_2.4_1609687217108.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("translate_en_mr", lang = "xx") 

result = pipeline.annotate("Your sentence to translate!")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("translate_en_mr", lang = "xx")

val result = pipeline.annotate("Your sentence to translate!")
```

{:.nlu-block}
```python
import nlu

text = ["Your sentence to translate!"]
translate_df = nlu.load('xx.en.translate_to.mr').predict(text, output_level='sentence')
translate_df
```

</div>

## Results

```bash
+------------------------------+--------------------------+
|sentence                      |translation               |
+------------------------------+--------------------------+
|Your sentence to translate!   |तू तुझ्या वाक्याचा अनुवाद करशील!  | 
+------------------------------+--------------------------+
```


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|translate_en_mr|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Language:|xx|

## Data Source

[https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models)