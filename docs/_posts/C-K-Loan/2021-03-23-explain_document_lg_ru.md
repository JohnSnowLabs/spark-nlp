---
layout: model
title: Explain Document pipeline for Russian (explain_document_lg)
author: John Snow Labs
name: explain_document_lg
date: 2021-03-23
tags: [open_source, russian, explain_document_lg, pipeline, ru]
task: [Named Entity Recognition, Lemmatization]
language: ru
edition: Spark NLP 3.0.0
spark_version: 3.0
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_lg is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
        and recognizes entities .
         It performs most of the common text processing tasks on your dataframe

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_ru_3.0.0_3.0_1616501405939.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_lg', lang = 'ru')
annotations =  pipeline.fullAnnotate(""Здравствуйте из Джона Снежных Лабораторий! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_lg", lang = "ru")
val result = pipeline.fullAnnotate("Здравствуйте из Джона Снежных Лабораторий! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Здравствуйте из Джона Снежных Лабораторий! ""]
result_df = nlu.load('ru.explain.lg').predict(text)
result_df
    
```
</div>

## Results

```bash
|    | document                                        | sentence                                       | token                                                      | lemma                                                      | pos                                        | embeddings                   | ner                                   | entities                       |
|---:|:------------------------------------------------|:-----------------------------------------------|:-----------------------------------------------------------|:-----------------------------------------------------------|:-------------------------------------------|:-----------------------------|:--------------------------------------|:-------------------------------|
|  0 | ['Здравствуйте из Джона Снежных Лабораторий! '] | ['Здравствуйте из Джона Снежных Лабораторий!'] | ['Здравствуйте', 'из', 'Джона', 'Снежных', 'Лабораторий!'] | ['здравствовать', 'из', 'Джон', 'Снежных', 'Лабораторий!'] | ['NOUN', 'ADP', 'PROPN', 'PROPN', 'PROPN'] | [[0.0, 0.0, 0.0, 0.0,.,...]] | ['O', 'O', 'B-PER', 'I-PER', 'I-PER'] | ['Джона Снежных Лабораторий!'] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_lg|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|