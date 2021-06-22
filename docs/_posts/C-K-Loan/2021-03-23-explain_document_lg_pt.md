---
layout: model
title: Explain Document pipeline for Portuguese (explain_document_lg)
author: John Snow Labs
name: explain_document_lg
date: 2021-03-23
tags: [open_source, portuguese, explain_document_lg, pipeline, pt]
supported: true
task: [Named Entity Recognition, Lemmatization]
language: pt
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_pt_3.0.0_3.0_1616505297906.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_lg', lang = 'pt')
annotations =  pipeline.fullAnnotate(""Olá de John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_lg", lang = "pt")
val result = pipeline.fullAnnotate("Olá de John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Olá de John Snow Labs! ""]
result_df = nlu.load('pt.explain.lg').predict(text)
result_df
    
```
</div>

## Results

```bash
|    | document                    | sentence                   | token                                  | lemma                                  | pos                                         | embeddings                   | ner                                   | entities            |
|---:|:----------------------------|:---------------------------|:---------------------------------------|:---------------------------------------|:--------------------------------------------|:-----------------------------|:--------------------------------------|:--------------------|
|  0 | ['Olá de John Snow Labs! '] | ['Olá de John Snow Labs!'] | ['Olá', 'de', 'John', 'Snow', 'Labs!'] | ['Olá', 'de', 'John', 'Snow', 'Labs!'] | ['PROPN', 'ADP', 'PROPN', 'PROPN', 'PROPN'] | [[0.4388400018215179,.,...]] | ['O', 'O', 'B-PER', 'I-PER', 'I-PER'] | ['John Snow Labs!'] |
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
|Language:|pt|