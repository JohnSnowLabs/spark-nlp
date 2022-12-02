---
layout: model
title: Basic NLP Pipeline for Spanish from TEMU_BSC for PlanTL
author: cayorodriguez
name: pipeline_bsc_roberta_base_bne
date: 2022-11-22
tags: [es, open_source]
task: Pipeline Public
language: es
edition: Spark NLP 4.0.0
spark_version: 3.2
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Basic NLP pipeline,  by TEMU-BSC for PlanTL-GOB-ES, with Tokenization, lemmatization, NER, embeddings and Normalization, using roberta_base_bne transformer.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/cayorodriguez/pipeline_bsc_roberta_base_bne_es_4.0.0_3.2_1669122787149.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

import sparknlp
spark = sparknlp.start()

from sparknlp.annotator import *
from sparknlp.base import *
pipeline = PretrainedPipeline("pipeline_bsc_roberta_base_bne", "es", "@cayorodriguez")
from sparknlp.base import LightPipeline

light_model = LightPipeline(pipeline)
text = "La Reserva Federal de el Gobierno de EE UU aprueba una de las mayorores subidas de tipos de interés desde 1994."
light_result = light_model.annotate(text)


result = pipeline.annotate(""Veo al hombre de los Estados Unidos con el telescopio"")

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
spark = sparknlp.start()

from sparknlp.annotator import *
from sparknlp.base import *
pipeline = PretrainedPipeline("pipeline_bsc_roberta_base_bne", "es", "@cayorodriguez")
from sparknlp.base import LightPipeline

light_model = LightPipeline(pipeline)
text = "La Reserva Federal de el Gobierno de EE UU aprueba una de las mayorores subidas de tipos de interés desde 1994."
light_result = light_model.annotate(text)


result = pipeline.annotate(""Veo al hombre de los Estados Unidos con el telescopio"")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_bsc_roberta_base_bne|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Community|
|Language:|es|
|Size:|2.0 GB|
|Dependencies:|roberta_base_bne|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- NormalizerModel
- StopWordsCleaner
- RoBertaEmbeddings
- SentenceEmbeddings
- EmbeddingsFinisher
- LemmatizerModel
- RoBertaForTokenClassification
- RoBertaForTokenClassification
- NerConverter