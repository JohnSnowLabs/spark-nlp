---
layout: model
title: German Bert Embeddings (Large, Cased)
author: John Snow Labs
name: bert_embeddings_gbert_large
date: 2022-04-11
tags: [bert, embeddings, de, open_source]
task: Embeddings
language: de
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `gbert-large` is a German model orginally trained by `deepset`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_gbert_large_de_3.4.2_3.0_1649676005002.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_gbert_large","de") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Ich liebe Funken NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_gbert_large","de") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Ich liebe Funken NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.embed.gbert_large").predict("""Ich liebe Funken NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_gbert_large|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|de|
|Size:|1.3 GB|
|Case sensitive:|true|

## References

- https://huggingface.co/deepset/gbert-large
- https://arxiv.org/pdf/2010.10906.pdf
- https://arxiv.org/pdf/2010.10906.pdf
- https://deepset.ai/german-bert
- https://deepset.ai/germanquad
- https://github.com/deepset-ai/FARM
- https://github.com/deepset-ai/haystack/
- https://twitter.com/deepset_ai
- https://www.linkedin.com/company/deepset-ai/
- https://haystack.deepset.ai/community/join
- https://github.com/deepset-ai/haystack/discussions
- https://deepset.ai
- http://www.deepset.ai/jobs