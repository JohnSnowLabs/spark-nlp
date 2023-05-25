---
layout: model
title: Recognize Entities DL pipeline for Italian - Large
author: John Snow Labs
name: entity_recognizer_lg
date: 2023-05-24
tags: [open_source, italian, entity_recognizer_lg, pipeline, it]
task: Named Entity Recognition
language: it
edition: Spark NLP 4.4.2
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The entity_recognizer_lg is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
        and recognizes entities .
         It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_it_4.4.2_3.4_1684941483648.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_it_4.4.2_3.4_1684941483648.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('entity_recognizer_lg', lang = 'it')
annotations =  pipeline.fullAnnotate(""Ciao da John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("entity_recognizer_lg", lang = "it")
val result = pipeline.fullAnnotate("Ciao da John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Ciao da John Snow Labs! ""]
result_df = nlu.load('it.ner.lg').predict(text)
result_df
    
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('entity_recognizer_lg', lang = 'it')
annotations =  pipeline.fullAnnotate(""Ciao da John Snow Labs! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("entity_recognizer_lg", lang = "it")
val result = pipeline.fullAnnotate("Ciao da John Snow Labs! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Ciao da John Snow Labs! ""]
result_df = nlu.load('it.ner.lg').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                     | sentence                    | token                                   | embeddings                   | ner                                   | entities            |
|---:|:-----------------------------|:----------------------------|:----------------------------------------|:-----------------------------|:--------------------------------------|:--------------------|
|  0 | ['Ciao da John Snow Labs! '] | ['Ciao da John Snow Labs!'] | ['Ciao', 'da', 'John', 'Snow', 'Labs!'] | [[-0.238279998302459,.,...]] | ['O', 'O', 'I-PER', 'I-PER', 'I-PER'] | ['John Snow Labs!'] |


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|entity_recognizer_lg|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|2.5 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter