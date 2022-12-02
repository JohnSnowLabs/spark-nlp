---
layout: model
title: Arabic Bert Embeddings (Base, Arabert Model)
author: John Snow Labs
name: bert_embeddings_bert_base_arabert
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

Pretrained Bert Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-base-arabert` is a Arabic model orginally trained by `aubmindlab`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_bert_base_arabert_ar_3.4.2_3.0_1649677303708.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_arabert","ar") \
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

val embeddings = BertEmbeddings.pretrained("bert_embeddings_bert_base_arabert","ar") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("أنا أحب شرارة NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ar.embed.bert_base_arabert").predict("""أنا أحب شرارة NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_bert_base_arabert|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ar|
|Size:|507.5 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/aubmindlab/bert-base-arabert
- https://github.com/google-research/bert
- https://arxiv.org/abs/2003.00104
- https://github.com/WissamAntoun/pydata_khobar_meetup
- http://alt.qcri.org/farasa/segmenter.html
- /aubmindlab/bert-base-arabert/resolve/main/(https://github.com/google-research/bert/blob/master/multilingual.md)
- https://github.com/elnagara/HARD-Arabic-Dataset
- https://www.aclweb.org/anthology/D15-1299
- https://staff.aub.edu.lb/~we07/Publications/ArSentD-LEV_Sentiment_Corpus.pdf
- https://github.com/mohamedadaly/LABR
- http://curtis.ml.cmu.edu/w/courses/index.php/ANERcorp
- https://github.com/husseinmozannar/SOQAL
- https://github.com/aub-mind/arabert/blob/master/AraBERT/README.md
- https://arxiv.org/abs/2003.00104v2
- https://archive.org/details/arwiki-20190201
- https://www.semanticscholar.org/paper/1.5-billion-words-Arabic-Corpus-El-Khair/f3eeef4afb81223df96575adadf808fe7fe440b4
- https://www.aclweb.org/anthology/W19-4619
- https://sites.aub.edu.lb/mindlab/
- https://www.yakshof.com/#/
- https://www.behance.net/rahalhabib
- https://www.linkedin.com/in/wissam-antoun-622142b4/
- https://twitter.com/wissam_antoun
- https://github.com/WissamAntoun
- https://www.linkedin.com/in/fadybaly/
- https://twitter.com/fadybaly
- https://github.com/fadybaly