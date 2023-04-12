---
layout: model
title: Recognize Entities DL Pipeline for French - Medium
author: John Snow Labs
name: entity_recognizer_md
date: 2021-03-22
tags: [open_source, french, entity_recognizer_md, pipeline, fr]
supported: true
task: [Named Entity Recognition, Lemmatization, Part of Speech Tagging]
language: fr
edition: Spark NLP 3.0.0
spark_version: 3.0
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The entity_recognizer_md is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps.
It performs most of the common text processing tasks on your dataframe

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/2da56c087da53a2fac1d51774d49939e05418e57/jupyter/annotation/english/explain-document-dl/Explain%20Document%20DL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_fr_3.0.0_3.0_1616445658156.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/entity_recognizer_md_fr_3.0.0_3.0_1616445658156.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('entity_recognizer_md', lang = 'fr')
annotations =  pipeline.fullAnnotate(""Bonjour de John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("entity_recognizer_md", lang = "fr")
val result = pipeline.fullAnnotate("Bonjour de John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Bonjour de John Snow Labs! ""]
result_df = nlu.load('fr.ner.md').predict(text)
result_df

```
</div>

## Results

```bash
|    | document                        | sentence                       | token                                      | embeddings                   | ner                                        | entities                       |
|---:|:--------------------------------|:-------------------------------|:-------------------------------------------|:-----------------------------|:-------------------------------------------|:-------------------------------|
|  0 | ['Bonjour de John Snow Labs! '] | ['Bonjour de John Snow Labs!'] | ['Bonjour', 'de', 'John', 'Snow', 'Labs!'] | [[0.0783179998397827,.,...]] | ['I-MISC', 'O', 'I-PER', 'I-PER', 'I-PER'] | ['Bonjour', 'John Snow Labs!'] |
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
|Language:|fr|
