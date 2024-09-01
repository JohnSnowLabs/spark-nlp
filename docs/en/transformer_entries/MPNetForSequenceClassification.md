{%- capture title -%}
MPNetForSequenceClassification
{%- endcapture -%}

{%- capture description -%}
MPNetForSequenceClassification can load MPNet Models with sequence classification/regression
head on top (a linear layer on top of the pooled output) e.g. for multi-class document
classification tasks.

Note that currently, only SetFit models can be imported.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val sequenceClassifier = MPNetForSequenceClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
```

The default model is `"mpnet_sequence_classifier_ukr_message"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Text+Classification).

To see which models are compatible and how to import them see
[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669) and to see more extended
examples, see
[MPNetForSequenceClassificationTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MPNetForSequenceClassificationTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

sequenceClassifier = MPNetForSequenceClassification \
    .pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("label")

data = spark.createDataFrame([
    ["I love driving my car."],
    ["The next bus will arrive in 20 minutes."],
    ["pineapple on pizza is the worst 🤮"],
]).toDF("text")

pipeline = Pipeline().setStages([document, tokenizer, sequenceClassifier])
pipelineModel = pipeline.fit(data)
results = pipelineModel.transform(data)
results.select("label.result").show()
+--------------------+
|              result|
+--------------------+
|     [TRANSPORT/CAR]|
|[TRANSPORT/MOVEMENT]|
|              [FOOD]|
+--------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val sequenceClassifier = MPNetForSequenceClassification
  .pretrained()
  .setInputCols(Array("document", "token"))
  .setOutputCol("label")

val texts = Seq(
  "I love driving my car.",
  "The next bus will arrive in 20 minutes.",
  "pineapple on pizza is the worst 🤮")
val data = texts.toDF("text")

val pipeline = new Pipeline().setStages(Array(document, tokenizer, sequenceClassifier))
val pipelineModel = pipeline.fit(data)
val results = pipelineModel.transform(data)

results.select("label.result").show()
+--------------------+
|              result|
+--------------------+
|     [TRANSPORT/CAR]|
|[TRANSPORT/MOVEMENT]|
|              [FOOD]|
+--------------------+

{%- endcapture -%}

{%- capture api_link -%}
[MPNetForSequenceClassification](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/MPNetForSequenceClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[MPNetForSequenceClassification](/api/python/reference/autosummary/sparknlp/annotator/classifier_dl/mpnet_for_sequence_classification/index.html#sparknlp.annotator.classifier_dl.mpnet_for_sequence_classification.MPNetForSequenceClassification)
{%- endcapture -%}

{%- capture source_link -%}
[MPNetForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MPNetForSequenceClassification.scala)
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