---
layout: model
title: English RoBERTa Embeddings (Smiles Strings, v1)
author: John Snow Labs
name: roberta_embeddings_chEMBL_smiles_v1
date: 2022-04-14
tags: [roberta, embeddings, en, open_source]
task: Embeddings
language: en
nav_key: models
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `chEMBL_smiles_v1` is a English model orginally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_chEMBL_smiles_v1_en_3.4.2_3.0_1649947021342.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_chEMBL_smiles_v1_en_3.4.2_3.0_1649947021342.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols("document") \
.setOutputCol("token")

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_chEMBL_smiles_v1","en") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_chEMBL_smiles_v1","en") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.chEMBL_smiles_v1").predict("""I love Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_chEMBL_smiles_v1|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|88.8 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/mrm8488/chEMBL_smiles_v1
- https://onlinelibrary.wiley.com/doi/full/10.1002/minf.201700111
- https://github.com/topazape/LSTM_Chem/blob/master/cleanup_smiles.py
- https://github.com/topazape/LSTM_Chem
- https://www.ncbi.nlm.nih.gov/pubmed/29095571
- https://twitter.com/mrm8488
- https://www.linkedin.com/in/manuel-romero-cs/