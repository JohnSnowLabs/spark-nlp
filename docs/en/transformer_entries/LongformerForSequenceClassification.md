{%- capture title -%}
LongformerForSequenceClassification
{%- endcapture -%}

{%- capture description -%}
LongformerForSequenceClassification can load Longformer Models with sequence classification/regression head on top
(a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val sequenceClassifier = LongformerForSequenceClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
```
The default model is `"longformer_base_sequence_classifier_imdb"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Text+Classification).

Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. The Spark NLP Workshop
example shows how to import them https://github.com/JohnSnowLabs/spark-nlp/discussions/5669.
and the [LongformerForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/LongformerForSequenceClassificationTestSpec.scala).
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

sequenceClassifier = LongformerForSequenceClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCaseSensitive(True)

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    sequenceClassifier
])

data = spark.createDataFrame([[\"\"\"John Lenon was born in London and lived
in Paris. My name is Sarah and I live in London\"\"\"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("label.result").show(truncate=False)
+--------------------+
|result              |
+--------------------+
|[neg, neg]          |
|[pos, pos, pos, pos]|
+--------------------+
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

val sequenceClassifier = LongformerForSequenceClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  sequenceClassifier
))

val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("label.result").show(false)
+--------------------+
|result              |
+--------------------+
|[neg, neg]          |
|[pos, pos, pos, pos]|
+--------------------+

{%- endcapture -%}

{%- capture api_link -%}
[LongformerForSequenceClassification](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/LongformerForSequenceClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[LongformerForSequenceClassification](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/longformer_for_sequence_classification/index.html#sparknlp.annotator.classifier_dl.longformer_for_sequence_classification.LongformerForSequenceClassification)
{%- endcapture -%}

{%- capture source_link -%}
[LongformerForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/LongformerForSequenceClassification.scala)
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