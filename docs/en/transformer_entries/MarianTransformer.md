{%- capture title -%}
MarianTransformer
{%- endcapture -%}

{%- capture description -%}
MarianTransformer: Fast Neural Machine Translation

Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies.
It is mainly being developed by the Microsoft Translator team. Many academic (most notably the University of
Edinburgh and in the past the Adam Mickiewicz University in Poznań) and commercial contributors help with its
development. MarianTransformer uses the models trained by MarianNMT.

It is currently the engine behind the Microsoft Translator Neural Machine Translation services and being deployed by
many companies, organizations and research projects.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val marian = MarianTransformer.pretrained()
  .setInputCols("sentence")
  .setOutputCol("translation")
```
The default model is `"opus_mt_en_fr"`, default language is `"xx"` (meaning multi-lingual), if no values are provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Translation).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/TRANSLATION_MARIAN.ipynb)
and the [MarianTransformerTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/MarianTransformerTestSpec.scala).

**Sources** :

[MarianNMT at GitHub](https://marian-nmt.github.io/)

[Marian: Fast Neural Machine Translation in C++ ](https://www.aclweb.org/anthology/P18-4020/)

**Paper Abstract:**

*We present Marian, an efficient and self-contained Neural Machine Translation framework with an integrated
automatic differentiation engine based on dynamic computation graphs. Marian is written entirely in C++. We describe
the design of the encoder-decoder framework and demonstrate that a research-friendly toolkit can achieve high
training and translation speed.*

**Note:**

This is a very computationally expensive module especially on larger sequence.
The use of an accelerator such as GPU is recommended.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \
    .setInputCols("document") \
    .setOutputCol("sentence")

marian = MarianTransformer.pretrained() \
    .setInputCols("sentence") \
    .setOutputCol("translation") \
    .setMaxInputLength(30)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      sentence,
      marian
    ])

data = spark.createDataFrame([["What is the capital of France? We should know this in french."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(translation.result) as result").show(truncate=False)
+-------------------------------------+
|result                               |
+-------------------------------------+
|Quelle est la capitale de la France ?|
|On devrait le savoir en français.    |
+-------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
  .setInputCols("document")
  .setOutputCol("sentence")

val marian = MarianTransformer.pretrained()
  .setInputCols("sentence")
  .setOutputCol("translation")
  .setMaxInputLength(30)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentence,
    marian
  ))

val data = Seq("What is the capital of France? We should know this in french.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(translation.result) as result").show(false)
+-------------------------------------+
|result                               |
+-------------------------------------+
|Quelle est la capitale de la France ?|
|On devrait le savoir en français.    |
+-------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[MarianTransformer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/seq2seq/MarianTransformer)
{%- endcapture -%}

{%- capture python_api_link -%}
[MarianTransformer](/api/python/reference/autosummary/python/sparknlp/annotator/seq2seq/marian_transformer/index.html#sparknlp.annotator.seq2seq.marian_transformer.MarianTransformer)
{%- endcapture -%}

{%- capture source_link -%}
[MarianTransformer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/MarianTransformer.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}