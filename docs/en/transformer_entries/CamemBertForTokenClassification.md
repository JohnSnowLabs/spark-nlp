{%- capture title -%}
CamemBertForTokenClassification
{%- endcapture -%}

{%- capture description -%}
CamemBertForTokenClassification can load CamemBERT Models with a token classification head on
top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition
(NER) tasks.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val tokenClassifier = CamemBertForTokenClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
```
The default model is `"camembert_base_token_classifier_wikiner"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition).

and the
[CamemBertForTokenClassificationTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/CamemBertForTokenClassificationTestSpec.scala).
To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \\
    .setInputCol("text") \\
    .setOutputCol("document")
tokenizer = Tokenizer() \\
    .setInputCols(["document"]) \\
    .setOutputCol("token")
tokenClassifier = CamemBertForTokenClassification.pretrained() \\
    .setInputCols(["token", "document"]) \\
    .setOutputCol("label") \\
    .setCaseSensitive(True)
  
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    tokenClassifier
])

data = spark.createDataFrame([["george washington est allé à washington"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("label.result").show(truncate=False)
+------------------------------+
|result                        |
+------------------------------+
|[I-PER, I-PER, O, O, O, I-LOC]|
+------------------------------+

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

val tokenClassifier = CamemBertForTokenClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
  .setCaseSensitive(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  tokenClassifier
))

val data = Seq("george washington est allé à washington").toDF("text")
val result = pipeline.fit(data).transform(data)

result.select("label.result").show(false)
+------------------------------+
|result                        |
+------------------------------+
|[I-PER, I-PER, O, O, O, I-LOC]|
+------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[CamemBertForTokenClassification](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/CamemBertForTokenClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[CamemBertForTokenClassification](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/camembert_for_token_classification/index.html#sparknlp.annotator.classifier_dl.camembert_for_token_classification.CamemBertForTokenClassification)
{%- endcapture -%}

{%- capture source_link -%}
[CamemBertForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/CamemBertForTokenClassification.scala)
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