---
layout: model
title: RoBERTa base biomedical
author: ireneisdoomed
name: roberta_base_biomedical
date: 2022-01-13
tags: [es, open_source]
task: Text Classification
language: es
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: false
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model has been pulled from the HF Hub - https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es

This is a result of reproducing the tutorial for bringing HF's models into Spark NLP - https://medium.com/spark-nlp/importing-huggingface-models-into-sparknlp-8c63bdea671d

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/ireneisdoomed/roberta_base_biomedical_es_3.4.0_3.0_1642093372752.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("term")\
.setOutputCol("document")

tokenizer = Tokenizer()\
.setInputCols("document")\
.setOutputCol("token")

roberta_embeddings = RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es", "@ireneisdoomed")\
.setInputCols(["document", "token"])\
.setOutputCol("roberta_embeddings")

pipeline = Pipeline(stages = [
documentAssembler,
tokenizer,
roberta_embeddings])
```



{:.nlu-block}
```python
import nlu
nlu.load("es.embed.roberta_base_biomedical").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_biomedical|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|es|
|Size:|301.7 MB|
