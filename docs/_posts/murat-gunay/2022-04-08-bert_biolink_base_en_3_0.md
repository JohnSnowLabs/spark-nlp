---
layout: model
title: BERT Base Biolink Embeddings
author: John Snow Labs
name: bert_biolink_base
date: 2022-04-08
tags: [bert, medical, biolink, en, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This embeddings component was trained on PubMed abstracts all along with citations link information. The embeddings were introduced in this [paper](https://arxiv.org/abs/2203.15827). This model achieves state-of-the-art performance on several biomedical NLP benchmarks such as [BLURB](https://microsoft.github.io/BLURB/) and [MedQA-USMLE](https://github.com/jind11/MedQA).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_biolink_base_en_3.4.2_3.0_1649419433513.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_biolink_base", "en")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_biolink_base", "en")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.e").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_biolink_base|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|406.4 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[https://pubmed.ncbi.nlm.nih.gov/](https://pubmed.ncbi.nlm.nih.gov/)

```
@InProceedings{yasunaga2022linkbert,
author =  {Michihiro Yasunaga and Jure Leskovec and Percy Liang},
title =   {LinkBERT: Pretraining Language Models with Document Links},
year =    {2022},  
booktitle = {Association for Computational Linguistics (ACL)},  
}
```

## Benchmarking

```bash
Scores for several benchmark datasets :

- BLURB : 83.39
- PubMedQA : 70.2
- BioASQ : 91.4
- MedQA-USMLE : 40.0
```
