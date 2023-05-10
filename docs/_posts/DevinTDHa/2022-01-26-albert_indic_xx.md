---
layout: model
title: IndicBERT - Albert for 12 major Indian languages
author: John Snow Labs
name: albert_indic
date: 2022-01-26
tags: [open_source, albert, as, bn, en, gu, kn, ml, mr, or, pa, ta, te, xx]
task: Embeddings
language: xx
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: AlBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IndicBERT is a multilingual ALBERT model pretrained exclusively on 12 major Indian languages. It is pre-trained on our novel monolingual corpus of around 9 billion tokens and subsequently evaluated on a set of diverse tasks. IndicBERT has much fewer parameters than other multilingual models (mBERT, XLM-R etc.) while it also achieves a performance on-par or better than these models.

The 12 languages covered by IndicBERT are: Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_indic_xx_3.4.0_3.0_1643211494926.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_indic_xx_3.4.0_3.0_1643211494926.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")

embeddings = AlbertEmbeddings.pretrained("albert_indic","xx") \
.setInputCols(["document",'token'])\
.setOutputCol("embeddings")\

embeddingsFinisher = EmbeddingsFinisher() \
.setInputCols(["embeddings"]) \
.setOutputCols("finished_embeddings") \
.setOutputAsVector(True) \
.setCleanAnnotations(False)

pipeline = Pipeline().setStages([
documentAssembler,
tokenizer,
embeddings,
embeddingsFinisher
])

data = spark.createDataFrame([
["கர்நாடக சட்டப் பேரவையில் வெற்றி பெற்ற எம்எல்ஏக்கள் இன்று பதவியேற்றுக் கொண்ட நிலையில் , காங்கிரஸ் எம்எல்ஏ ஆனந்த் சிங் க்கள் ஆப்சென்ட் ஆகி அதிர்ச்சியை ஏற்படுத்தியுள்ளார் . உச்சநீதிமன்ற உத்தரவுப்படி இன்று மாலை முதலமைச்சர் எடியூரப்பா இன்று நம்பிக்கை வாக்கெடுப்பு நடத்தி பெரும்பான்மையை நிரூபிக்க உச்சநீதிமன்றம் உத்தரவிட்டது ."],
]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val embeddings = AlbertEmbeddings.pretrained("albert_indic", "xx")
.setInputCols("token", "document")
.setOutputCol("embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
.setInputCols("embeddings")
.setOutputCols("finished_embeddings")
.setOutputAsVector(true)
.setCleanAnnotations(false)

val pipeline = new Pipeline().setStages(Array(
documentAssembler,
tokenizer,
embeddings,
embeddingsFinisher
))

val data = Seq("கர்நாடக சட்டப் பேரவையில் வெற்றி பெற்ற எம்எல்ஏக்கள் இன்று பதவியேற்றுக் கொண்ட நிலையில் , காங்கிரஸ் எம்எல்ஏ ஆனந்த் சிங் க்கள் ஆப்சென்ட் ஆகி அதிர்ச்சியை ஏற்படுத்தியுள்ளார் . உச்சநீதிமன்ற உத்தரவுப்படி இன்று மாலை முதலமைச்சர் எடியூரப்பா இன்று நம்பிக்கை வாக்கெடுப்பு நடத்தி பெரும்பான்மையை நிரூபிக்க உச்சநீதிமன்றம் உத்தரவிட்டது .")
.toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
```


{:.nlu-block}
```python
import nlu
nlu.load("xx.embed.albert.indic").predict("""கர்நாடக சட்டப் பேரவையில் வெற்றி பெற்ற எம்எல்ஏக்கள் இன்று பதவியேற்றுக் கொண்ட நிலையில் , காங்கிரஸ் எம்எல்ஏ ஆனந்த் சிங் க்கள் ஆப்சென்ட் ஆகி அதிர்ச்சியை ஏற்படுத்தியுள்ளார் . உச்சநீதிமன்ற உத்தரவுப்படி இன்று மாலை முதலமைச்சர் எடியூரப்பா இன்று நம்பிக்கை வாக்கெடுப்பு நடத்தி பெரும்பான்மையை நிரூபிக்க உச்சநீதிமன்றம் உத்தரவிட்டது .""")
```

</div>

## Results

```bash
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.2693195641040802,-0.6446362733840942,-0.05138964205980301,0.06030936539173...|
|[0.027906809002161026,-0.37459731101989746,-0.08371371030807495,-0.0869174525...|
|[0.3804604113101959,-0.7870151400566101,0.08463867008686066,-0.30186718702316...|
|[0.15204764902591705,-0.26839596033096313,0.07375998795032501,-0.131638795137...|
|[0.1482795625925064,-0.221298485994339,-0.022987276315689087,-0.2132280170917...|
+--------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_indic|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Size:|128.3 MB|

## References

The model was exported from transformers and is based on https://github.com/AI4Bharat/indic-bert