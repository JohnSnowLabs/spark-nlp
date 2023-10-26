{%- capture title -%}
DeBertaForTokenClassification
{%- endcapture -%}

{%- capture description -%}
DeBertaForTokenClassification can load DeBERTA Models v2 and v3 with a token classification
head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val tokenClassifier = DeBertaForTokenClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
```

The default model is `"deberta_v3_xsmall_token_classifier_conll03"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Named+Entity+Recognition).

and the
[DeBertaForTokenClassificationTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/DeBertaForTokenClassificationTestSpec.scala).
Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669.
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture python_example -%}
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
tokenClassifier = DeBertaForTokenClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCaseSensitive(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    tokenClassifier
])

data = spark.createDataFrame([["John Lenon was born in London and lived in Paris. My name is Sarah and I live in London"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("label.result").show(truncate=False)
+------------------------------------------------------------------------------------+
|result                                                                              |
+------------------------------------------------------------------------------------+
|[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
+------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val tokenClassifier = DeBertaForTokenClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  tokenClassifier
))

val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("label.result").show(false)
+------------------------------------------------------------------------------------+
|result                                                                              |
+------------------------------------------------------------------------------------+
|[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
+------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[DeBertaForTokenClassification](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/DeBertaForTokenClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[DeBertaForTokenClassification](/api/python/reference/autosummary/sparknlp/annotator/classifier_dl/deberta_for_token_classification/index.html#sparknlp.annotator.classifier_dl.deberta_for_token_classification.DeBertaForTokenClassification)
{%- endcapture -%}

{%- capture source_link -%}
[DeBertaForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/DeBertaForTokenClassification.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}