---
layout: model
title: Recognize Entities DL Pipeline for Polish - Small
author: John Snow Labs
name: entity_recognizer_sm
date: 2023-05-25
tags: [open_source, polish, entity_recognizer_sm, pipeline, pl]
task: Named Entity Recognition
language: pl
edition: Spark NLP 4.4.2
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The entity_recognizer_sm is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps.
         It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_pl_4.4.2_3.4_1685049478793.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/entity_recognizer_sm_pl_4.4.2_3.4_1685049478793.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('entity_recognizer_sm', lang = 'pl')
annotations =  pipeline.fullAnnotate(""Witaj z John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("entity_recognizer_sm", lang = "pl")
val result = pipeline.fullAnnotate("Witaj z John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Witaj z John Snow Labs! ""]
result_df = nlu.load('pl.ner').predict(text)
result_df
    
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('entity_recognizer_sm', lang = 'pl')
annotations =  pipeline.fullAnnotate(""Witaj z John Snow Labs! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("entity_recognizer_sm", lang = "pl")
val result = pipeline.fullAnnotate("Witaj z John Snow Labs! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Witaj z John Snow Labs! ""]
result_df = nlu.load('pl.ner').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                     | sentence                    | token                                   | embeddings                   | ner                                   | entities            |
|---:|:-----------------------------|:----------------------------|:----------------------------------------|:-----------------------------|:--------------------------------------|:--------------------|
|  0 | ['Witaj z John Snow Labs! '] | ['Witaj z John Snow Labs!'] | ['Witaj', 'z', 'John', 'Snow', 'Labs!'] | [[0.0, 0.0, 0.0, 0.0,.,...]] | ['O', 'O', 'B-PER', 'I-PER', 'I-PER'] | ['John Snow Labs!'] |


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|entity_recognizer_sm|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|pl|
|Size:|167.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter