---
layout: model
title: CamemBERT Base Wikipedia
author: John Snow Labs
name: camembert_base_wikipedia_4gb
date: 2022-05-04
tags: [fr, french, embeddings, camembert, wikipedia, open_source]
task: Embeddings
language: fr
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: CamemBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

[CamemBERT](https://arxiv.org/abs/1911.03894) is a state-of-the-art language model for French based on the RoBERTa model.
For further information or requests, please go to [Camembert Website](https://camembert-model.fr/)

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_base_wikipedia_4gb_fr_3.4.4_3.0_1651673934156.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camembert_base_wikipedia_4gb_fr_3.4.4_3.0_1651673934156.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = CamemBertEmbeddings.pretrained("camembert_base_wikipedia_4gb", "fr") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = CamemBertEmbeddings.pretrained("camembert_base_wikipedia_4gb", "fr")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("fr.embed.camembert_wiki_4g").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_base_wikipedia_4gb|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|fr|
|Size:|265.5 MB|
|Case sensitive:|true|

## References

[https://huggingface.co/camembert/camembert-base-wikipedia-4gb](https://huggingface.co/camembert/camembert-base-wikipedia-4gb)

## Benchmarking

```bash
| Model                          | #params                        | Arch. | Training data                     |
|--------------------------------|--------------------------------|-------|-----------------------------------|
| `camembert-base` | 110M   | Base  | OSCAR (138 GB of text)            |
| `camembert/camembert-large`              | 335M    | Large | CCNet (135 GB of text)            |
| `camembert/camembert-base-ccnet`         | 110M    | Base  | CCNet (135 GB of text)            |
| `camembert/camembert-base-wikipedia-4gb` | 110M    | Base  | Wikipedia (4 GB of text)          |
| `camembert/camembert-base-oscar-4gb`     | 110M    | Base  | Subsample of OSCAR (4 GB of text) |
| `camembert/camembert-base-ccnet-4gb`     | 110M    | Base  | Subsample of CCNet (4 GB of text) |
```