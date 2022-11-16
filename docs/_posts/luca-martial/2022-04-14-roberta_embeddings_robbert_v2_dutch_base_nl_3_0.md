---
layout: model
title: Dutch RoBERTa Embeddings
author: John Snow Labs
name: roberta_embeddings_robbert_v2_dutch_base
date: 2022-04-14
tags: [roberta, embeddings, nl, open_source]
task: Embeddings
language: nl
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
recommended: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `robbert-v2-dutch-base` is a Dutch model orginally trained by `pdelobelle`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_robbert_v2_dutch_base_nl_3.4.2_3.0_1649949003731.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_robbert_v2_dutch_base","nl") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Ik hou van vonk nlp"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_robbert_v2_dutch_base","nl") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Ik hou van vonk nlp").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("nl.embed.robbert_v2_dutch_base").predict("""Ik hou van vonk nlp""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_robbert_v2_dutch_base|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|nl|
|Size:|438.6 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/pdelobelle/robbert-v2-dutch-base
- https://github.com/iPieter/RobBERT
- https://scholar.google.com/scholar?oi=bibs&hl=en&cites=7180110604335112086
- https://www.aclweb.org/anthology/2021.wassa-1.27/
- https://arxiv.org/pdf/2001.06286.pdf
- https://biblio.ugent.be/publication/8704637/file/8704638.pdf
- https://arxiv.org/pdf/2001.06286.pdf
- https://arxiv.org/pdf/2001.06286.pdf
- https://arxiv.org/pdf/2004.02814.pdf
- https://github.com/proycon/deepfrog
- https://arxiv.org/pdf/2001.06286.pdf
- https://github.com/proycon/deepfrog
- https://arxiv.org/pdf/2001.06286.pdf
- https://arxiv.org/pdf/2010.13652.pdf
- https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/automatic-classification-of-participant-roles-in-cyberbullying-can-we-detect-victims-bullies-and-bystanders-in-social-media-text/A2079C2C738C29428E666810B8903342
- https://gitlab.com/spelfouten/dutch-simpletransformers/
- https://arxiv.org/pdf/2101.05716.pdf
- https://medium.com/broadhorizon-cmotions/nlp-with-r-part-5-state-of-the-art-in-nlp-transformers-bert-3449e3cd7494
- https://people.cs.kuleuven.be/~pieter.delobelle/robbert/
- https://arxiv.org/abs/2001.06286
- https://github.com/iPieter/RobBERT
- https://arxiv.org/abs/1907.11692
- https://github.com/pytorch/fairseq/tree/master/examples/roberta
- https://people.cs.kuleuven.be/~pieter.delobelle/robbert/
- https://arxiv.org/abs/2001.06286
- https://github.com/iPieter/RobBERT
- https://github.com/benjaminvdb/110kDBRD
- https://www.statmt.org/europarl/
- https://arxiv.org/abs/2001.02943
- https://universaldependencies.org/treebanks/nl_lassysmall/index.html
- https://www.clips.uantwerpen.be/conll2002/ner/
- https://oscar-corpus.com/
- https://github.com/pytorch/fairseq/tree/master/examples/roberta
- https://github.com/pytorch/fairseq/tree/master/examples/roberta
- https://arxiv.org/abs/2001.06286
- https://github.com/iPieter/RobBERT#how-to-replicate-our-paper-experiments
- https://arxiv.org/abs/1909.11942
- https://camembert-model.fr/
- https://en.wikipedia.org/wiki/Robbert
- https://muppet.fandom.com/wiki/Bert
- https://github.com/iPieter/RobBERT/blob/master/res/robbert_logo.png
- https://people.cs.kuleuven.be/~pieter.delobelle
- https://thomaswinters.be
- https://people.cs.kuleuven.be/~bettina.berendt/
