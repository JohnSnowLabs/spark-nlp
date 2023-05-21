---
layout: model
title: Recognize Entities OntoNotes pipeline - BERT Mini
author: John Snow Labs
name: onto_recognize_entities_bert_mini
date: 2023-05-21
tags: [open_source, english, onto_recognize_entities_bert_mini, pipeline, en]
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

The onto_recognize_entities_bert_mini is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
and recognizes entities .
It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/onto_recognize_entities_bert_mini_en_4.4.2_3.0_1684646004112.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/onto_recognize_entities_bert_mini_en_4.4.2_3.0_1684646004112.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('onto_recognize_entities_bert_mini', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("onto_recognize_entities_bert_mini", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.ner.onto.bert.mini').predict(text)
result_df

```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('onto_recognize_entities_bert_mini', lang = 'en')
annotations =  pipeline.fullAnnotate(""Hello from John Snow Labs ! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("onto_recognize_entities_bert_mini", lang = "en")
val result = pipeline.fullAnnotate("Hello from John Snow Labs ! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Hello from John Snow Labs ! ""]
result_df = nlu.load('en.ner.onto.bert.mini').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                         | sentence                        | token                                          | embeddings                   | ner                                        | entities           |
|---:|:---------------------------------|:--------------------------------|:-----------------------------------------------|:-----------------------------|:-------------------------------------------|:-------------------|
|  0 | ['Hello from John Snow Labs ! '] | ['Hello from John Snow Labs !'] | ['Hello', 'from', 'John', 'Snow', 'Labs', '!'] | [[-0.147406503558158,.,...]] | ['O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O'] | ['John Snow Labs'] |


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|onto_recognize_entities_bert_mini|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|57.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- NerDLModel
- NerConverter