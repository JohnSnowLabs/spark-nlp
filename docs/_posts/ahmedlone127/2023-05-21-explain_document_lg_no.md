---
layout: model
title: Explain Document pipeline for Norwegian (Bokmal) (explain_document_lg)
author: John Snow Labs
name: explain_document_lg
date: 2023-05-21
tags: [open_source, norwegian_bokmal, explain_document_lg, pipeline, "no"]
task: Named Entity Recognition
language: "no"
edition: Spark NLP 4.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_lg is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
and recognizes entities .
It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_no_4.4.2_3.0_1684639461035.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_lg_no_4.4.2_3.0_1684639461035.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_lg', lang = 'no')
annotations =  pipeline.fullAnnotate(""Hei fra John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_lg", lang = "no")
val result = pipeline.fullAnnotate("Hei fra John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Hei fra John Snow Labs! ""]
result_df = nlu.load('no.explain.lg').predict(text)
result_df

```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_lg', lang = 'no')
annotations =  pipeline.fullAnnotate(""Hei fra John Snow Labs! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("explain_document_lg", lang = "no")
val result = pipeline.fullAnnotate("Hei fra John Snow Labs! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Hei fra John Snow Labs! ""]
result_df = nlu.load('no.explain.lg').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                     | sentence                    | token                                   | lemma                                   | pos                                         | embeddings                   | ner                                    | entities               |
|---:|:-----------------------------|:----------------------------|:----------------------------------------|:----------------------------------------|:--------------------------------------------|:-----------------------------|:---------------------------------------|:-----------------------|
|  0 | ['Hei fra John Snow Labs! '] | ['Hei fra John Snow Labs!'] | ['Hei', 'fra', 'John', 'Snow', 'Labs!'] | ['Hei', 'fra', 'John', 'Snow', 'Labs!'] | ['PROPN', 'ADP', 'PROPN', 'PROPN', 'PROPN'] | [[0.0639619976282119,.,...]] | ['O', 'O', 'B-PER', 'I-PER', 'B-PROD'] | ['John Snow', 'Labs!'] |


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_lg|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|no|
|Size:|2.5 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- LemmatizerModel
- PerceptronModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter