---
layout: model
title: Recognize Entities DL Pipeline for Finnish - Medium
author: John Snow Labs
name: entity_recognizer_md
date: 2021-03-22
tags: [open_source, finnish, entity_recognizer_md, pipeline, fi]
supported: true
task: [Named Entity Recognition, Lemmatization]
language: fi
edition: Spark NLP 3.0.0
spark_version: 3.0
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The entity_recognizer_md is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
and recognizes entities .
It performs most of the common text processing tasks on your dataframe

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_fi_3.0.0_3.0_1616456428015.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('entity_recognizer_md', lang = 'fi')
annotations =  pipeline.fullAnnotate(""Hei John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("entity_recognizer_md", lang = "fi")
val result = pipeline.fullAnnotate("Hei John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hei John Snow Labs! ""]
result_df = nlu.load('fi.ner.md').predict(text)
result_df

```
</div>

## Results

```bash
|    | document                 | sentence                | token                            | embeddings                   | ner                              | entities            |
|---:|:-------------------------|:------------------------|:---------------------------------|:-----------------------------|:---------------------------------|:--------------------|
|  0 | ['Hei John Snow Labs! '] | ['Hei John Snow Labs!'] | ['Hei', 'John', 'Snow', 'Labs!'] | [[0.1868100017309188,.,...]] | ['O', 'B-PER', 'I-PER', 'I-PER'] | ['John Snow Labs!'] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|entity_recognizer_md|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fi|