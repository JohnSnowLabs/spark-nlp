---
layout: model
title: Arabic Bert Embeddings (Base, Trained on an eighth of the full MSA dataset)
author: John Snow Labs
name: bert_embeddings_bert_base_arabic_camelbert_msa_eighth
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

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-base-arabic-camelbert-msa-eighth` is a Arabic model orginally trained by `CAMeL-Lab`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bert_base_arabic_camelbert_msa_eighth_ar_3.4.2_3.0_1649678829456.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_bert_base_arabic_camelbert_msa_eighth_ar_3.4.2_3.0_1649678829456.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_arabic_camelbert_msa_eighth","ar") \
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

val embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_arabic_camelbert_msa_eighth","ar") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("أنا أحب شرارة NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ar.embed.bert_base_arabic_camelbert_msa_eighth").predict("""أنا أحب شرارة NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_bert_base_arabic_camelbert_msa_eighth|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ar|
|Size:|409.3 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa-eighth
- https://arxiv.org/abs/2103.06678
- https://github.com/CAMeL-Lab/CAMeLBERT
- https://catalog.ldc.upenn.edu/LDC2011T11
- http://www.abuelkhair.net/index.php/en/arabic/abu-el-khair-corpus
- https://vlo.clarin.eu/search;jsessionid=31066390B2C9E8C6304845BA79869AC1?1&q=osian
- https://archive.org/details/arwiki-20190201
- https://oscar-corpus.com/
- https://github.com/google-research/bert
- https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L286-L297
- https://github.com/CAMeL-Lab/camel_tools
- https://github.com/CAMeL-Lab/CAMeLBERT