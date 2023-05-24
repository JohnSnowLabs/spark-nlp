---
layout: model
title: Explain Document Pipeline for French
author: John Snow Labs
name: explain_document_md
date: 2023-05-24
tags: [open_source, french, explain_document_md, pipeline, fr]
task: Named Entity Recognition
language: fr
edition: Spark NLP 4.4.2
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_md is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps.
It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_fr_4.4.2_3.4_1684939620146.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_md_fr_4.4.2_3.4_1684939620146.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_md', lang = 'fr')
annotations =  pipeline.fullAnnotate(""Bonjour de John Snow Labs! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_md", lang = "fr")
val result = pipeline.fullAnnotate("Bonjour de John Snow Labs! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Bonjour de John Snow Labs! ""]
result_df = nlu.load('fr.explain.md').predict(text)
result_df

```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_md', lang = 'fr')
annotations =  pipeline.fullAnnotate(""Bonjour de John Snow Labs! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("explain_document_md", lang = "fr")
val result = pipeline.fullAnnotate("Bonjour de John Snow Labs! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Bonjour de John Snow Labs! ""]
result_df = nlu.load('fr.explain.md').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                        | sentence                       | token                                      | lemma                                      | pos                                        | embeddings                   | ner                                        | entities                       |
|---:|:--------------------------------|:-------------------------------|:-------------------------------------------|:-------------------------------------------|:-------------------------------------------|:-----------------------------|:-------------------------------------------|:-------------------------------|
|  0 | ['Bonjour de John Snow Labs! '] | ['Bonjour de John Snow Labs!'] | ['Bonjour', 'de', 'John', 'Snow', 'Labs!'] | ['Bonjour', 'de', 'John', 'Snow', 'Labs!'] | ['INTJ', 'ADP', 'PROPN', 'PROPN', 'PROPN'] | [[0.0783179998397827,.,...]] | ['I-MISC', 'O', 'I-PER', 'I-PER', 'I-PER'] | ['Bonjour', 'John Snow Labs!'] |


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_md|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|467.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- LemmatizerModel
- PerceptronModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter