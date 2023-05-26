---
layout: model
title: Explain Document Pipeline for Spanish
author: John Snow Labs
name: explain_document_sm
date: 2023-05-25
tags: [open_source, spanish, explain_document_sm, pipeline, es]
task: Named Entity Recognition
language: es
edition: Spark NLP 4.4.2
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_sm is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps.
It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_es_4.4.2_3.4_1685043872827.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_sm_es_4.4.2_3.4_1685043872827.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_sm', lang = 'es')
annotations =  pipeline.fullAnnotate(""Hola de John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_sm", lang = "es")
val result = pipeline.fullAnnotate("Hola de John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hola de John Snow Labs! ""]
result_df = nlu.load('es.explain').predict(text)
result_df

```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_sm', lang = 'es')
annotations =  pipeline.fullAnnotate(""Hola de John Snow Labs! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("explain_document_sm", lang = "es")
val result = pipeline.fullAnnotate("Hola de John Snow Labs! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Hola de John Snow Labs! ""]
result_df = nlu.load('es.explain').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                     | sentence                    | token                                   | lemma                                   | pos                                        | embeddings                   | ner                                    | entities               |
|---:|:-----------------------------|:----------------------------|:----------------------------------------|:----------------------------------------|:-------------------------------------------|:-----------------------------|:---------------------------------------|:-----------------------|
|  0 | ['Hola de John Snow Labs! '] | ['Hola de John Snow Labs!'] | ['Hola', 'de', 'John', 'Snow', 'Labs!'] | ['Hola', 'de', 'John', 'Snow', 'Labs!'] | ['PART', 'ADP', 'PROPN', 'PROPN', 'PROPN'] | [[0.1754499971866607,.,...]] | ['O', 'O', 'B-PER', 'I-PER', 'B-MISC'] | ['John Snow', 'Labs!'] |


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_sm|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|177.4 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- LemmatizerModel
- PerceptronModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter