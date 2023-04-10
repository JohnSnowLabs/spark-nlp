{%- capture title -%}
CamemBertForSequenceClassification
{%- endcapture -%}

{%- capture description -%}
CamemBertForSequenceClassification can load CamemBERT Models with sequence
classification/regression head on top (a linear layer on top of the pooled output) e.g. for
multi-class document classification tasks.

Pretrained models can be loaded with `pretrained` of the companion object:

```
val sequenceClassifier = CamemBertForSequenceClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
```

The default model is `camembert_base_sequence_classifier_allocine"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Text+Classification).

To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[CamemBertForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/CamemBertForSequenceClassificationTestSpec.scala).
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
sequenceClassifier = CamemBertForSequenceClassification.pretrained() \
    .setInputCols(["token", "document"]) \
    .setOutputCol("label") \
    .setCaseSensitive(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    sequenceClassifier
])
data = spark.createDataFrame([["j'ai adoré ce film lorsque j'étais enfant.", "Je déteste ça."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("class.result").show(truncate=False)
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

val sequenceClassifier = CamemBertForSequenceClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  sequenceClassifier
))

val data = Seq("j'ai adoré ce film lorsque j'étais enfant.", "Je déteste ça.").toDF("text")
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
[CamemBertForSequenceClassification](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/CamemBertForSequenceClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[CamemBertForSequenceClassification](/api/python/reference/autosummary/sparknlp/annotator/classifier_dl/camembert_for_token_classification/index.html#sparknlp.annotator.classifier_dl.camembert_for_sequence_classification.CamemBertForSequenceClassification)
{%- endcapture -%}

{%- capture source_link -%}
[CamemBertForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/CamemBertForSequenceClassification.scala)
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