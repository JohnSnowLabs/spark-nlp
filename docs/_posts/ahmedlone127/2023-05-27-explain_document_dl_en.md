---
layout: model
title: Explain Document DL Pipeline for English
author: John Snow Labs
name: explain_document_dl
date: 2023-05-27
tags: [open_source, english, explain_document_dl, pipeline, en]
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

The explain_document_dl is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps 
and recognizes entities .
It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_4.4.2_3.2_1685186531034.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_dl_en_4.4.2_3.2_1685186531034.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
Results


+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------+-----------+
|                                              text|                                          document|                                          sentence|                                             token|                                           checked|                                             lemma|                                              stem|                                               pos|                                        embeddings|                                         ner|   entities|
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------+-----------+
|The Mona Lisa is an oil painting from the 16th ...|[The Mona Lisa is an oil painting from the 16th...|[The Mona Lisa is an oil painting from the 16th...|[The, Mona, Lisa, is, an, oil, painting, from, ...|[The, Mona, Lisa, is, an, oil, painting, from, ...|[The, Mona, Lisa, be, an, oil, painting, from, ...|[the, mona, lisa, i, an, oil, paint, from, the,...|[DT, NNP, NNP, VBZ, DT, NN, NN, IN, DT, JJ, NN, .]|[[-0.038194, -0.24487, 0.72812, -0.39961, 0.083...|[O, B-PER, I-PER, O, O, O, O, O, O, O, O, O]|[Mona Lisa]|
+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------------+--------------------------------------------+-----------+


{:.model-param}
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
|Size:|176.2 MB|

## Included Models

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