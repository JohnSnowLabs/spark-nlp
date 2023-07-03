---
layout: model
title: Explain Document Pipeline for Danish
author: John Snow Labs
name: explain_document_sm
date: 2023-05-27
tags: [open_source, danish, explain_document_sm, pipeline, da]
task: Named Entity Recognition
language: da
edition: Spark NLP 4.4.2
spark_version: 3.2
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_sm_da_4.4.2_3.2_1685199771386.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_sm_da_4.4.2_3.2_1685199771386.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_sm', lang = 'da')
annotations =  pipeline.fullAnnotate(""Hej fra John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_sm", lang = "da")
val result = pipeline.fullAnnotate("Hej fra John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hej fra John Snow Labs! ""]
result_df = nlu.load('da.explain').predict(text)
result_df

```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_sm', lang = 'da')
annotations =  pipeline.fullAnnotate(""Hej fra John Snow Labs! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("explain_document_sm", lang = "da")
val result = pipeline.fullAnnotate("Hej fra John Snow Labs! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Hej fra John Snow Labs! ""]
result_df = nlu.load('da.explain').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                     | sentence                    | token                                   | lemma                                   | pos                                        | embeddings                   | ner                                   | entities            |
|---:|:-----------------------------|:----------------------------|:----------------------------------------|:----------------------------------------|:-------------------------------------------|:-----------------------------|:--------------------------------------|:--------------------|
|  0 | ['Hej fra John Snow Labs! '] | ['Hej fra John Snow Labs!'] | ['Hej', 'fra', 'John', 'Snow', 'Labs!'] | ['Hej', 'fra', 'John', 'Snow', 'Labs!'] | ['NOUN', 'ADP', 'PROPN', 'PROPN', 'PROPN'] | [[0.0306969992816448,.,...]] | ['O', 'O', 'B-PER', 'I-PER', 'I-PER'] | ['John Snow Labs!'] |


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
|Language:|da|
|Size:|168.4 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- LemmatizerModel
- PerceptronModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter