---
layout: model
title: Recognize Entities DL pipeline for French - Large
author: John Snow Labs
name: entity_recognizer_lg
date: 2023-05-27
tags: [open_source, french, entity_recognizer_lg, pipeline, fr]
task: Named Entity Recognition
language: fr
edition: Spark NLP 4.4.2
spark_version: 3.2
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_fr_4.4.2_3.2_1685183509564.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_fr_4.4.2_3.2_1685183509564.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('entity_recognizer_lg', lang = 'fr')
annotations =  pipeline.fullAnnotate(""Bonjour de John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("entity_recognizer_lg", lang = "fr")
val result = pipeline.fullAnnotate("Bonjour de John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Bonjour de John Snow Labs! ""]
result_df = nlu.load('fr.ner').predict(text)
result_df

```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('entity_recognizer_lg', lang = 'fr')
annotations =  pipeline.fullAnnotate(""Bonjour de John Snow Labs! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("entity_recognizer_lg", lang = "fr")
val result = pipeline.fullAnnotate("Bonjour de John Snow Labs! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Bonjour de John Snow Labs! ""]
result_df = nlu.load('fr.ner').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                        | sentence                       | token                                      | embeddings                   | ner                                   | entities            |
|---:|:--------------------------------|:-------------------------------|:-------------------------------------------|:-----------------------------|:--------------------------------------|:--------------------|
|  0 | ['Bonjour de John Snow Labs! '] | ['Bonjour de John Snow Labs!'] | ['Bonjour', 'de', 'John', 'Snow', 'Labs!'] | [[-0.010997000150382,.,...]] | ['O', 'O', 'I-PER', 'I-PER', 'I-PER'] | ['John Snow Labs!'] |


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
|Language:|fr|
|Size:|2.5 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter