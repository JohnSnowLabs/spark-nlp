---
layout: model
title: Explain Document DL Pipeline for English
author: John Snow Labs
name: explain_document_dl
date: 2021-03-23
tags: [open_source, english, explain_document_dl, pipeline, en]
supported: true
task: [Named Entity Recognition, Lemmatization]
language: en
nav_key: models
edition: Spark NLP 3.0.0
spark_version: 3.0
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_dl is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
and recognizes entities .
It performs most of the common text processing tasks on your dataframe

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_EN_18/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_3.0.0_3.0_1616473268265.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_3.0.0_3.0_1616473268265.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline('explain_document_dl', lang = 'en')
annotations =  pipeline.fullAnnotate("The Mona Lisa is an oil painting from the 16th century.")[0]
annotations.keys()

```
```scala

val pipeline = new PretrainedPipeline("explain_document_dl", lang = "en")
val result = pipeline.fullAnnotate("The Mona Lisa is an oil painting from the 16th century.")(0)


```

{:.nlu-block}
```python

import nlu
text = ["The Mona Lisa is an oil painting from the 16th century."]
result_df = nlu.load('en.explain.dl').predict(text)
result_df

```
</div>

## Results

```bash
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------+-----------+
|                                              text|                                          document|                                          sentence|                                             token|                                           checked|                                             lemma|                                              stem|                                               pos|                                        embeddings|                                         ner|   entities|
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------+-----------+
|The Mona Lisa is an oil painting from the 16th ...|[The Mona Lisa is an oil painting from the 16th...|[The Mona Lisa is an oil painting from the 16th...|[The, Mona, Lisa, is, an, oil, painting, from, ...|[The, Mona, Lisa, is, an, oil, painting, from, ...|[The, Mona, Lisa, be, an, oil, painting, from, ...|[the, mona, lisa, i, an, oil, paint, from, the,...|[DT, NNP, NNP, VBZ, DT, NN, NN, IN, DT, JJ, NN, .]|[[-0.038194, -0.24487, 0.72812, -0.39961, 0.083...|[O, B-PER, I-PER, O, O, O, O, O, O, O, O, O]|[Mona Lisa]|
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_dl|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
