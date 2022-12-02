---
layout: model
title: Explain Document ML Pipeline for English
author: John Snow Labs
name: explain_document_ml
date: 2021-03-23
tags: [open_source, english, explain_document_ml, pipeline, en]
supported: true
task: [Named Entity Recognition, Lemmatization]
language: en
edition: Spark NLP 3.0.0
spark_version: 3.0
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_ml is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
and recognizes entities .
It performs most of the common text processing tasks on your dataframe

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_ml_en_3.0.0_3.0_1616473253101.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline('explain_document_ml', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_ml", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.explain').predict(text)
result_df

```
</div>

## Results

```bash
|    | document                         | sentence                         | token                                            | spell                                           | lemmas                                          | stems                                          | pos                                    |
|---:|:---------------------------------|:---------------------------------|:-------------------------------------------------|:------------------------------------------------|:------------------------------------------------|:-----------------------------------------------|:---------------------------------------|
|  0 | ['Hello fronm John Snwow Labs!'] | ['Hello fronm John Snwow Labs!'] | ['Hello', 'fronm', 'John', 'Snwow', 'Labs', '!'] | ['Hello', 'front', 'John', 'Snow', 'Labs', '!'] | ['Hello', 'front', 'John', 'Snow', 'Labs', '!'] | ['hello', 'front', 'john', 'snow', 'lab', '!'] | ['UH', 'NN', 'NNP', 'NNP', 'NNP', '.'] ||    | document   | sentence   | token     | spell     | lemmas    | stems     | pos    |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_ml|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
