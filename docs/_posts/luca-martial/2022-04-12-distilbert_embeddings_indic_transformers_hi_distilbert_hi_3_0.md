---
layout: model
title: Hindi DistilBERT Embeddings (from neuralspace-reverie)
author: John Snow Labs
name: distilbert_embeddings_indic_transformers_hi_distilbert
date: 2022-04-12
tags: [distilbert, embeddings, hi, open_source]
task: Embeddings
language: hi
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: DistilBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `indic-transformers-hi-distilbert` is a Hindi model orginally trained by `neuralspace-reverie`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_indic_transformers_hi_distilbert_hi_3.4.2_3.0_1649783440506.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_embeddings_indic_transformers_hi_distilbert_hi_3.4.2_3.0_1649783440506.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_indic_transformers_hi_distilbert","hi") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["मुझे स्पार्क एनएलपी पसंद है"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = DistilBertEmbeddings.pretrained("distilbert_embeddings_indic_transformers_hi_distilbert","hi") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("मुझे स्पार्क एनएलपी पसंद है").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("hi.embed.indic_transformers_hi_distilbert").predict("""मुझे स्पार्क एनएलपी पसंद है""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_embeddings_indic_transformers_hi_distilbert|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|hi|
|Size:|247.6 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/neuralspace-reverie/indic-transformers-hi-distilbert
- https://oscar-corpus.com/