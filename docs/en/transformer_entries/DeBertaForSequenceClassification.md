{%- capture title -%}
DeBertaForSequenceClassification
{%- endcapture -%}

{%- capture description -%}
DeBertaForSequenceClassification can load DeBerta v2 & v3 Models with sequence
classification/regression head on top (a linear layer on top of the pooled output) e.g. for
multi-class document classification tasks.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val sequenceClassifier = DeBertaForSequenceClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
```

The default model is `"deberta_v3_xsmall_sequence_classifier_imdb"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Text+Classification).

To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[DeBertaForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/DeBertaForSequenceClassificationTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN, DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
CATEGORY
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
sequenceClassifier = DeBertaForSequenceClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCaseSensitive(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame([["I loved this movie when I was a child.", "It was pretty boring."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("label.result").show(truncate=False)
+------+
|result|
+------+
|[pos] |
|[neg] |
+------+
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

val sequenceClassifier = DeBertaForSequenceClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  sequenceClassifier
))

val data = Seq("I loved this movie when I was a child.", "It was pretty boring.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("label.result").show(false)
+------+
|result|
+------+
|[pos] |
|[neg] |
+------+

{%- endcapture -%}

{%- capture api_link -%}
[DeBertaForSequenceClassification](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/DeBertaForSequenceClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[DeBertaForSequenceClassification](/api/python/reference/autosummary/sparknlp/annotator/classifier_dl/deberta_for_sequence_classification/index.html#sparknlp.annotator.classifier_dl.deberta_for_sequence_classification.DeBertaForSequenceClassification)
{%- endcapture -%}

{%- capture source_link -%}
[DeBertaForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/DeBertaForSequenceClassification.scala)
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