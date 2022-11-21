---
layout: model
title: Spell Checking Pipeline for English
author: John Snow Labs
name: check_spelling
date: 2021-03-26
tags: [open_source, english, check_spelling, pipeline, en]
supported: true
task: [Spell Check,]
language: en
edition: Spark NLP 3.0.0
spark_version: 3.0
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The check_spelling is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
        and recognizes entities .
         It performs most of the common text processing tasks on your dataframe

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SPELL_CHECKER_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SPELL_CHECKER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/check_spelling_en_3.0.0_3.0_1616772629811.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('check_spelling', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("check_spelling", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('').predict(text)
result_df
    
```
</div>

## Results

```bash
|    | document                         | sentence                        | token                                          | checked                                        |
|---:|:---------------------------------|:--------------------------------|:-----------------------------------------------|:-----------------------------------------------|
|  0 | ['I liek to live dangertus ! '] | ['I liek to live dangertus !'] | ['I', 'liek', 'to', 'live', 'dangertus', '!'] | ['I', 'like', 'to', 'live', 'dangerous', '!'] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|check_spelling|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- NorvigSweetingModel
