{%- capture title -%}
BertForZeroShotClassification
{%- endcapture -%}

{%- capture description -%}
BertForZeroShotClassification using a `ModelForSequenceClassification` trained on NLI (natural
language inference) tasks. Equivalent of `BertForSequenceClassification` models, but these
models don't require a hardcoded number of potential classes, they can be chosen at runtime.
It usually means it's slower but it is much more flexible.

Note that the model will loop through all provided labels. So the more labels you have, the
longer this process will take.

Any combination of sequences and labels can be passed and each combination will be posed as a
premise/hypothesis pair and passed to the pretrained model.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val sequenceClassifier = BertForZeroShotClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
```

The default model is `"bert_base_cased_zero_shot_classifier_xnli"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Text+Classification).

To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[BertForZeroShotClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/BertForZeroShotClassification.scala).
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

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")
sequenceClassifier = BertForZeroShotClassification.pretrained() \
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

val sequenceClassifier = BertForZeroShotClassification.pretrained()
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
[BertForZeroShotClassification](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/BertForZeroShotClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[BertForZeroShotClassification](/api/python/reference/autosummary/sparknlp/annotator/classifier_dl/bert_for_zero_shot_classification/index.html#sparknlp.annotator.classifier_dl.bert_for_zero_shot_classification.BertForZeroShotClassification)
{%- endcapture -%}

{%- capture source_link -%}
[BertForZeroShotClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/BertForZeroShotClassification.scala)
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