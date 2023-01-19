---
layout: model
title: Arabic Bert Embeddings (MARBERT model v2)
author: John Snow Labs
name: bert_embeddings_MARBERTv2
date: 2022-04-11
tags: [bert, embeddings, ar, open_source]
task: Embeddings
language: ar
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `MARBERTv2` is a Arabic model orginally trained by `UBC-NLP`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_MARBERTv2_ar_3.4.2_3.0_1649678231280.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_MARBERTv2_ar_3.4.2_3.0_1649678231280.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_MARBERTv2","ar") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["أنا أحب شرارة NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_MARBERTv2","ar") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("أنا أحب شرارة NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ar.embed.MARBERTv2").predict("""أنا أحب شرارة NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_MARBERTv2|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ar|
|Size:|609.4 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/UBC-NLP/MARBERTv2
- https://aclanthology.org/2021.acl-long.551.pdf
- https://github.com/UBC-NLP/marbert
- https://doi.org/10.14288/SOCKEYE
- https://www.tensorflow.org/tfrc