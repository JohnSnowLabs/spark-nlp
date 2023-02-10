---
layout: model
title: Clinical Longformer
author: John Snow Labs
name: clinical_longformer
date: 2022-02-08
tags: [longformer, clinical, en, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: LongformerEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This embeddings model was imported from `Hugging Face`([link](https://huggingface.co/yikuan8/Clinical-Longformer)). Clinical-Longformer is a clinical knowledge enriched version of `Longformer` that was further pretrained using MIMIC-III clinical notes. It allows up to 4,096 tokens as the model input. 

`Clinical-Longformer` consistently out-performs `ClinicalBERT` across 10 baseline dataset for at least 2 percent. Those downstream experiments broadly cover named entity recognition (NER), question answering (QA), natural language inference (NLI) and text classification tasks.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/clinical_longformer_en_3.4.0_3.0_1644309598171.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/clinical_longformer_en_3.4.0_3.0_1644309598171.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = LongformerEmbeddings.pretrained("clinical_longformer", "en")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")\
.setCaseSensitive(True)\
.setMaxSentenceLength(4096)
```
```scala
val embeddings = LongformerEmbeddings.pretrained("clinical_longformer", "en")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
.setCaseSensitive(True)\
.setMaxSentenceLength(4096)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.longformer.clinical").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_longformer|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|534.9 MB|
|Case sensitive:|true|
|Max sentence length:|4096|

## References

[https://arxiv.org/pdf/2201.11838.pdf](https://arxiv.org/pdf/2201.11838.pdf)