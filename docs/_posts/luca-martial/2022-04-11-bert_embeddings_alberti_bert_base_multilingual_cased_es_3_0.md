---
layout: model
title: Spanish Bert Embeddings (from flax-community)
author: John Snow Labs
name: bert_embeddings_alberti_bert_base_multilingual_cased
date: 2022-04-11
tags: [bert, embeddings, es, open_source]
task: Embeddings
language: es
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `alberti-bert-base-multilingual-cased` is a Spanish model orginally trained by `flax-community`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_alberti_bert_base_multilingual_cased_es_3.4.2_3.0_1649671065273.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_alberti_bert_base_multilingual_cased","es") \
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

val embeddings = BertEmbeddings.pretrained("bert_embeddings_alberti_bert_base_multilingual_cased","es") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Me encanta chispa nlp").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.embed.alberti_bert_base_multilingual_cased").predict("""Me encanta chispa nlp""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_alberti_bert_base_multilingual_cased|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|es|
|Size:|667.2 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/flax-community/alberti-bert-base-multilingual-cased
- https://github.com/google/flax
- https://github.com/linhd-postdata/averell/
- https://postdata.linhd.uned.es/
- https://github.com/pruizf/disco
- https://github.com/bncolorado/CorpusSonetosSigloDeOro
- https://github.com/bncolorado/CorpusGeneralPoesiaLiricaCastellanaDelSigloDeOro
- https://github.com/linhd-postdata/gongocorpus
- http://obvil.sorbonne-universite.site/corpus/gongora/gongora_obra-poetica
- https://github.com/alhuber1502/ECPA
- https://github.com/waynegraham/for_better_for_verse
- https://crisco2.unicaen.fr/verlaine/index.php?navigation=accueil
- https://github.com/linhd-postdata/metrique-en-ligne
- https://github.com/linhd-postdata/biblioteca_italiana
- http://www.bibliotecaitaliana.it/
- https://github.com/versotym/corpusCzechVerse
- https://gitlab.com/stichotheque/stichotheque-pt
- https://github.com/linhd-postdata/poesi.as
- http://www.poesi.as/
- https://github.com/aparrish/gutenberg-poetry-corpus
- https://www.kaggle.com/ahmedabelal/arabic-poetry
- https://github.com/THUNLP-AIPoet/Datasets/tree/master/CCPC
- https://github.com/sks190/SKVR
- https://github.com/linhd-postdata/textgrid-poetry
- https://textgrid.de/en/digitale-bibliothek
- https://github.com/tnhaider/german-rhyme-corpus
- https://github.com/ELTE-DH/verskorpusz
- https://www.kaggle.com/oliveirasp6/poems-in-portuguese
- https://www.kaggle.com/grafstor/19-000-russian-poems
- https://discord.com/channels/858019234139602994/859113060068229190