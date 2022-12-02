{%- capture title -%}
RoBertaForTokenClassification
{%- endcapture -%}

{%- capture description -%}
RoBertaForTokenClassification can load RoBERTa Models with a token classification head on top (a linear layer on top of the hidden-states output)
e.g. for Named-Entity-Recognition (NER) tasks.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val tokenClassifier = RoBertaForTokenClassification.pretrained()
  .setInputCols("token", "document")
  .setOutputCol("label")
```
The default model is `"roberta_base_token_classifier_conll03"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition).

and the [RoBertaForTokenClassificationTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/RoBertaForTokenClassificationTestSpec.scala).
Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To see which models are compatible and how to import them see [Import Transformers into Spark NLP 🚀](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
NAMED_ENTITY
{%- endcapture -%}

{%- capture prediction_python_example -%}
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

tokenClassifier = RoBertaForTokenClassification.pretrained() \
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

{%- capture prediction_scala_example -%}
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

val tokenClassifier = RoBertaForTokenClassification.pretrained()
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

{%- capture training_python_example -%}
# This annotator needs to be trained externally. Please see the training page
# for instructions.
{%- endcapture -%}

{%- capture training_scala_example -%}
// This annotator needs to be trained externally. Please see the training page
// for instructions.
{%- endcapture -%}

{%- capture embeddings_python_example -%}
# This annotator has a fully connected layer attached for classification. For
# embeddings see the base transformer annotator.
{%- endcapture -%}

{%- capture embeddings_scala_example -%}
// This annotator has a fully connected layer attached for classification. For
// embeddings see the base transformer annotator.
{%- endcapture -%}

{%- capture api_link -%}
[RoBertaForTokenClassification](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/classifier/dl/RoBertaForTokenClassification)
{%- endcapture -%}

{%- capture python_api_link -%}
[RoBertaForTokenClassification](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/roberta_for_token_classification/index.html#sparknlp.annotator.classifier_dl.roberta_for_token_classification.RoBertaForTokenClassification)
{%- endcapture -%}

{%- capture source_link -%}
[RoBertaForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/RoBertaForTokenClassification.scala)
{%- endcapture -%}

{% include templates/transformer_usecases_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_api_link=python_api_link
api_link=api_link
source_link=source_link
prediction_python_example=prediction_python_example
prediction_scala_example=prediction_scala_example
training_python_example=training_python_example
training_scala_example=training_scala_example
embeddings_python_example=embeddings_python_example
embeddings_scala_example=embeddings_scala_example
%}