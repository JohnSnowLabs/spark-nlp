---
layout: model
title: BERT Biolink Embeddings
author: John Snow Labs
name: bert_biolink_large
date: 2022-04-08
tags: [biolink, bert, medical, en, open_source]
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

This embeddings component was trained on PubMed abstracts all along with citation link information. The model was introduced in [this paper](https://arxiv.org/abs/2203.15827), achieving state-of-the-art performance on several biomedical NLP benchmarks such as [BLURB](https://microsoft.github.io/BLURB/) and [MedQA-USMLE](https://github.com/jind11/MedQA).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_biolink_large_en_3.4.2_3.0_1649411676807.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_biolink_large_en_3.4.2_3.0_1649411676807.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_biolink_large", "en")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_biolink_large", "en")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.ge").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_biolink_large|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|1.3 GB|
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

- BLURB : 84.30
- PubMedQA : 72.2
- BioASQ : 94.8
- MedQA-USMLE : 44.6
```
