---
layout: model
title: Recognize Entities DL pipeline for English - BERT
author: John Snow Labs
name: recognize_entities_bert
date: 2023-05-20
tags: [open_source, english, recognize_entities_bert, pipeline, en]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The recognize_entities_bert is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
and recognizes entities .
It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/recognize_entities_bert_en_4.4.2_3.0_1684626456905.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/recognize_entities_bert_en_4.4.2_3.0_1684626456905.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('recognize_entities_bert', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("recognize_entities_bert", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.ner.bert').predict(text)
result_df

```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('recognize_entities_bert', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("recognize_entities_bert", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.ner.bert').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                         | sentence                        | token                                          | embeddings                   | ner                                        | entities           |
|---:|:---------------------------------|:--------------------------------|:-----------------------------------------------|:-----------------------------|:-------------------------------------------|:-------------------|
|  0 | ['Hello from John Snow Labs ! '] | ['Hello from John Snow Labs !'] | ['Hello', 'from', 'John', 'Snow', 'Labs', '!'] | [[-0.085488274693489,.,...]] | ['O', 'O', 'I-PER', 'I-PER', 'I-ORG', 'O'] | ['John Snow Labs'] ||    | document                         | sentence                        | token                                          | embeddings                   | ner                                        | entities           |



{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|recognize_entities_bert|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|424.5 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- BertEmbeddings
- NerDLModel
- NerConverter