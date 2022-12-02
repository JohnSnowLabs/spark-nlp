---
layout: model
title: Recognize Entities OntoNotes pipeline - ELECTRA Small
author: John Snow Labs
name: onto_recognize_entities_electra_small
date: 2021-03-22
tags: [open_source, english, onto_recognize_entities_electra_small, pipeline, en]
supported: true
task: [Named Entity Recognition, Lemmatization, Part of Speech Tagging]
language: en
edition: Spark NLP 3.0.0
spark_version: 3.0
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The onto_recognize_entities_electra_small is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps.
It performs most of the common text processing tasks on your dataframe

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/2da56c087da53a2fac1d51774d49939e05418e57/jupyter/annotation/english/explain-document-dl/Explain%20Document%20DL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_recognize_entities_electra_small_en_3.0.0_3.0_1616444187316.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('onto_recognize_entities_electra_small', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("onto_recognize_entities_electra_small", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.ner.onto.electra.small').predict(text)
result_df

```
</div>

## Results

```bash
|    | document                         | sentence                        | token                                          | embeddings                   | ner                                        | entities           |
|---:|:---------------------------------|:--------------------------------|:-----------------------------------------------|:-----------------------------|:-------------------------------------------|:-------------------|
|  0 | ['Hello from John Snow Labs ! '] | ['Hello from John Snow Labs !'] | ['Hello', 'from', 'John', 'Snow', 'Labs', '!'] | [[0.2279076874256134,.,...]] | ['O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O'] | ['John Snow Labs'] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_recognize_entities_electra_small|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|