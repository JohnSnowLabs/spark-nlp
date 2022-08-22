---
layout: model
title: Explain Document Pipeline for Russian
author: John Snow Labs
name: explain_document_md
date: 2021-03-22
tags: [open_source, russian, explain_document_md, pipeline, ru]
supported: true
task: [Named Entity Recognition, Lemmatization, Part of Speech Tagging]
language: ru
edition: Spark NLP 3.0.0
spark_version: 3.0
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_md is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps.
It performs most of the common text processing tasks on your dataframe

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/2da56c087da53a2fac1d51774d49939e05418e57/jupyter/annotation/english/explain-document-dl/Explain%20Document%20DL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_ru_3.0.0_3.0_1616432725151.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_md', lang = 'ru')
annotations =  pipeline.fullAnnotate(""Здравствуйте из Джона Снежных Лабораторий! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_md", lang = "ru")
val result = pipeline.fullAnnotate("Здравствуйте из Джона Снежных Лабораторий! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Здравствуйте из Джона Снежных Лабораторий! ""]
result_df = nlu.load('ru.explain.md').predict(text)
result_df

```
</div>

## Results

```bash
|    | document                                        | sentence                                       | token                                                      | lemma                                                      | pos                                        | embeddings                   | ner                                   | entities                       |
|---:|:------------------------------------------------|:-----------------------------------------------|:-----------------------------------------------------------|:-----------------------------------------------------------|:-------------------------------------------|:-----------------------------|:--------------------------------------|:-------------------------------|
|  0 | ['Здравствуйте из Джона Снежных Лабораторий! '] | ['Здравствуйте из Джона Снежных Лабораторий!'] | ['Здравствуйте', 'из', 'Джона', 'Снежных', 'Лабораторий!'] | ['здравствовать', 'из', 'Джон', 'Снежных', 'Лабораторий!'] | ['NOUN', 'ADP', 'PROPN', 'PROPN', 'PROPN'] | [[0.0, 0.0, 0.0, 0.0,.,...]] | ['O', 'O', 'B-LOC', 'I-LOC', 'I-LOC'] | ['Джона Снежных Лабораторий!'] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_md|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|