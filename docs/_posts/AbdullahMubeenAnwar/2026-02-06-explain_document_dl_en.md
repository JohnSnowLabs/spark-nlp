---
layout: model
title: Explain Document DL Pipeline for English
author: John Snow Labs
name: explain_document_dl
date: 2026-02-06
tags: [open_source, en, pipeline]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.4.2
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The `explain_document_dl` pipeline is a pretrained English NLP pipeline that handles basic text processing and named entity recognition. It can processes text and performs tasks like sentence detection, tokenization, spelling correction, lemmatization, stemming, part-of-speech tagging, word embeddings, and NER.

Included Models:
- DocumentAssembler
- SentenceDetector
- TokenizerModel
- NorvigSweetingModel
- LemmatizerModel
- Stemmer
- PerceptronModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_4.4.2_3.2_1770410351717.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_4.4.2_3.2_1770410351717.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline('explain_document_dl', lang = 'en')
annotations = pipeline.fullAnnotate("The Mona Lisa is an oil painting from the 16th century.")[0]
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("explain_document_dl", lang = "en")
val result = pipeline.fullAnnotate("The Mona Lisa is an oil painting from the 16th century.")(0)
```
</div>

## Results

```bash

+-------+-----+---+----+--------+------+
| token |begin|end| pos| lemma  | ner  |
+-------+-----+---+----+--------+------+
| The   |  0  |  2| DT | The    | O    |
| Mona  |  4  |  7| NNP| Mona   | B-PER|
| Lisa  |  9  | 12| NNP| Lisa   | I-PER|
| is    | 14  | 15| VBZ| be     | O    |
| an    | 17  | 18| DT | an     | O    |
| oil   | 20  | 22| NN | oil    | O    |
| painting|24 | 31| NN | painting| O   |
| from  | 33  | 36| IN | from   | O    |
| the   | 38  | 40| DT | the    | O    |
| 16th  | 42  | 45| JJ | 16th   | O    |
| century| 47 | 54| NN | century| O    |
+-------+-----+---+----+--------+------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_dl|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|176.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- RegexTokenizer
- NorvigSweetingModel
- LemmatizerModel
- Stemmer
- PerceptronModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter