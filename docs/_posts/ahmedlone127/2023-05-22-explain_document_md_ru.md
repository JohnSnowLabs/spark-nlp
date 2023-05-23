---
layout: model
title: Explain Document Pipeline for Russian
author: John Snow Labs
name: explain_document_md
date: 2023-05-22
tags: [open_source, russian, explain_document_md, pipeline, ru]
task: Named Entity Recognition
language: ru
edition: Spark NLP 4.4.2
spark_version: 3.2
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_md_ru_4.4.2_3.2_1684744687037.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_md_ru_4.4.2_3.2_1684744687037.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_md', lang = 'ru')
annotations =  pipeline.fullAnnotate(""Здравствуйте из Джона Снежных Лабораторий! "")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_md", lang = "ru")
val result = pipeline.fullAnnotate("Здравствуйте из Джона Снежных Лабораторий! ")(0)


```

{:.nlu-block}
```python

import nlu
text = [""Здравствуйте из Джона Снежных Лабораторий! ""]
result_df = nlu.load('ru.explain.md').predict(text)
result_df

```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipelinein
pipeline = PretrainedPipeline('explain_document_md', lang = 'ru')
annotations =  pipeline.fullAnnotate(""Здравствуйте из Джона Снежных Лабораторий! "")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("explain_document_md", lang = "ru")
val result = pipeline.fullAnnotate("Здравствуйте из Джона Снежных Лабораторий! ")(0)
```

{:.nlu-block}
```python
import nlu
text = [""Здравствуйте из Джона Снежных Лабораторий! ""]
result_df = nlu.load('ru.explain.md').predict(text)
result_df
```
</div>

## Results

```bash
Results


|    | document                                        | sentence                                       | token                                                      | lemma                                                      | pos                                        | embeddings                   | ner                                   | entities                       |
|---:|:------------------------------------------------|:-----------------------------------------------|:-----------------------------------------------------------|:-----------------------------------------------------------|:-------------------------------------------|:-----------------------------|:--------------------------------------|:-------------------------------|
|  0 | ['Здравствуйте из Джона Снежных Лабораторий! '] | ['Здравствуйте из Джона Снежных Лабораторий!'] | ['Здравствуйте', 'из', 'Джона', 'Снежных', 'Лабораторий!'] | ['здравствовать', 'из', 'Джон', 'Снежных', 'Лабораторий!'] | ['NOUN', 'ADP', 'PROPN', 'PROPN', 'PROPN'] | [[0.0, 0.0, 0.0, 0.0,.,...]] | ['O', 'O', 'B-LOC', 'I-LOC', 'I-LOC'] | ['Джона Снежных Лабораторий!'] |


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
|Language:|ru|
|Size:|465.5 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- LemmatizerModel
- PerceptronModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter