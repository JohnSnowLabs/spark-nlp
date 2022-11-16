---
layout: model
title: Portuguese Legal Bert Embeddings (Cased)
author: John Snow Labs
name: bert_embeddings_bert_base_cased_pt_lenerbr
date: 2022-04-11
tags: [bert, embeddings, pt, open_source]
task: Embeddings
language: pt
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
recommended: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Legal Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-base-cased-pt-lenerbr` is a Portuguese model orginally trained by `pierreguillou`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bert_base_cased_pt_lenerbr_pt_3.4.2_3.0_1649673986730.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_cased_pt_lenerbr","pt") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Eu amo Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_cased_pt_lenerbr","pt") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Eu amo Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("pt.embed.bert_base_cased_pt_lenerbr").predict("""Eu amo Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_bert_base_cased_pt_lenerbr|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|pt|
|Size:|408.8 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/pierreguillou/bert-base-cased-pt-lenerbr
- https://medium.com/@pierre_guillou/nlp-modelos-e-web-app-para-reconhecimento-de-entidade-nomeada-ner-no-dom%C3%ADnio-jur%C3%ADdico-b658db55edfb
- https://github.com/piegu/language-models/blob/master/Finetuning_language_model_BERtimbau_LeNER_Br.ipynb
- https://paperswithcode.com/sota?task=Fill+Mask&dataset=pierreguillou%2Flener_br_finetuning_language_model
