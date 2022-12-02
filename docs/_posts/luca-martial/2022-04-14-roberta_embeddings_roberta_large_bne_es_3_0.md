---
layout: model
title: Spanish RoBERTa Embeddings (Large)
author: John Snow Labs
name: roberta_embeddings_roberta_large_bne
date: 2022-04-14
tags: [roberta, embeddings, es, open_source]
task: Embeddings
language: es
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `roberta-large-bne` is a Spanish model orginally trained by `PlanTL-GOB-ES`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_roberta_large_bne_es_3.4.2_3.0_1649945069671.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_roberta_large_bne","es") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Me encanta chispa nlp"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_roberta_large_bne","es") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Me encanta chispa nlp").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.embed.roberta_large_bne").predict("""Me encanta chispa nlp""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_roberta_large_bne|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|es|
|Size:|848.5 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne
- https://arxiv.org/abs/1907.11692
- http://www.bne.es/en/Inicio/index.html
- http://www.bne.es/en/Inicio/index.html
- https://arxiv.org/abs/1907.11692
- https://github.com/PlanTL-GOB-ES/lm-spanish
- https://arxiv.org/abs/2107.07253