---
layout: model
title: Recognize Entities OntoNotes pipeline - BERT Large
author: John Snow Labs
name: onto_recognize_entities_bert_large
date: 2021-03-23
tags: [open_source, english, onto_recognize_entities_bert_large, pipeline, en]
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

The onto_recognize_entities_bert_large is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
and recognizes entities .
It performs most of the common text processing tasks on your dataframe

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_recognize_entities_bert_large_en_3.0.0_3.0_1616475201428.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/onto_recognize_entities_bert_large_en_3.0.0_3.0_1616475201428.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('onto_recognize_entities_bert_large', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("onto_recognize_entities_bert_large", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.ner.onto.bert.large').predict(text)
result_df

```
</div>

## Results

```bash
|    | document                         | sentence                        | token                                          | embeddings                   | ner                                        | entities           |
|---:|:---------------------------------|:--------------------------------|:-----------------------------------------------|:-----------------------------|:-------------------------------------------|:-------------------|
|  0 | ['Hello from John Snow Labs ! '] | ['Hello from John Snow Labs !'] | ['Hello', 'from', 'John', 'Snow', 'Labs', '!'] | [[-0.262016534805297,.,...]] | ['O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O'] | ['John Snow Labs'] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_recognize_entities_bert_large|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|