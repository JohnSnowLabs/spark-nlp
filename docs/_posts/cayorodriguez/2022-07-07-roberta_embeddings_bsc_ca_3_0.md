---
layout: model
title: Catalan RoBERTa embeddings
author: cayorodriguez
name: roberta_embeddings_bsc
date: 2022-07-07
tags: [roberta, projecte_aina, ca, open_source]
task: Embeddings
language: ca
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: false
recommended: true
annotator: RoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Catalan Roberta Word Embeddings, used within the `PlanTL-GOB-ES/roberta-base-ca` project. This model requires a specific Tokenizer, as shown in the Python Examples section.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/cayorodriguez/roberta_embeddings_bsc_ca_3.4.4_3.0_1657198648319.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

ex_list = ["aprox\.","pàg\.","p\.ex\.","gen\.","feb\.","abr\.","jul\.","set\.","oct\.","nov\.","dec\.","dr\.","dra\.","sr\.","sra\.","srta\.","núm\.","st\.","sta\.","pl\.","etc\.", "ex\."]
ex_list_all = []
ex_list_all.extend(ex_list)
ex_list_all.extend([x[0].upper() + x[1:] for x in ex_list])
ex_list_all.extend([x.upper() for x in ex_list])


tokenizer = Tokenizer() \
     .setInputCols(['sentence']).setOutputCol('token')\
     .setInfixPatterns(["(d|D)(els)","(d|D)(el)","(a|A)(ls)","(a|A)(l)","(p|P)(els)","(p|P)(el)",\
                            "([A-zÀ-ú_@]+)(-[A-zÀ-ú_@]+)",\
                             "(d'|D')([·A-zÀ-ú@_]+)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|'|,)+","(l'|L')([·A-zÀ-ú_]+)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|'|,)+", \
                             "(l'|l'|s'|s'|d'|d'|m'|m'|n'|n'|D'|D'|L'|L'|S'|S'|N'|N'|M'|M')([A-zÀ-ú_]+)",\
                             """([A-zÀ-ú·]+)(\.|,|\)|\?|!|;|\:|\"|”)(\.|,|\)|\?|!|;|\:|\"|”)+""",\
                             "([A-zÀ-ú·]+)('l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)(\.|,|;|:|\?|,)+",\
                             "([A-zÀ-ú·]+)('l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)",\
                             "(\.|\"|;|:|!|\?|\-|\(|\)|”|“|')+([0-9A-zÀ-ú_]+)",\
                             "([0-9A-zÀ-ú·]+)(\.|\"|;|:|!|\?|\(|\)|”|“|'|,|%)",\
                             "(\.|\"|;|:|!|\?|\-|\(|\)|”|“|,)+([0-9]+)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|,)+",\
                             "(d'|D'|l'|L')([·A-zÀ-ú@_]+)('l|'ns|'t|'m|'n|-les|-la|-lo|-li|-los|-me|-nos|-te|-vos|-se|-hi|-ne|-ho)(\.|\"|;|:|!|\?|\-|\(|\)|”|“|,)", \
                             "([\.|\"|;|:|!|\?|\-|\(|\)|”|“|,]+)([\.|\"|;|:|!|\?|\-|\(|\)|”|“|,]+)"]) \
         .setExceptions(ex_list_all).fit(data)
  
embeddings = WordEmbeddingsModel.pretrained("roberta_embeddings_bsc","ca") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["M'encanta fer anar aixó."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_bsc|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|ca|
|Size:|300.3 MB|
|Case sensitive:|true|

## References

projecte-aina/catalan_general_crawling @ huggingface
