{%- capture title -%}
CamemBertForQuestionAnswering
{%- endcapture -%}

{%- capture description -%}
CamemBertForQuestionAnswering can load CamemBERT Models with a span classification head on top
for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states
output to compute span start logits and span end logits).

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val spanClassifier = CamemBertForQuestionAnswering.pretrained()
  .setInputCols(Array("document_question", "document_context"))
  .setOutputCol("answer")
```

The default model is `"camembert_base_qa_fquad"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?task=Question+Answering).

To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
examples, see
[CamemBertForQuestionAnsweringTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/CamemBertForQuestionAnsweringTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = MultiDocumentAssembler() \
    .setInputCols(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])
spanClassifier = CamemBertForQuestionAnswering.pretrained() \
    .setInputCols(["document_question", "document_context"]) \
    .setOutputCol("answer") \
    .setCaseSensitive(False)
pipeline = Pipeline().setStages([
    documentAssembler,
    spanClassifier
])

data = spark.createDataFrame([["What's my name?", "My name is Clara and I live in Berkeley."]]).toDF("question", ontext")
result = pipeline.fit(data).transform(data)
result.select("answer.result").show(truncate=False)
+--------------------+
|result              |
+--------------------+
|[Clara]             |
+--------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = MultiDocumentAssembler() \
    .setInputCols(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])
spanClassifier = CamemBertForQuestionAnswering.pretrained() \
    .setInputCols(["document_question", "document_context"]) \
    .setOutputCol("answer") \
    .setCaseSensitive(False)
pipeline = Pipeline().setStages([
    documentAssembler,
    spanClassifier
])

data = spark.createDataFrame([["What's my name?", "My name is Clara and I live in Berkeley."]]).toDF("question", "context")
result = pipeline.fit(data).transform(data)
result.select("answer.result").show(truncate=False)
+--------------------+
|result              |
+--------------------+
|[Clara]             |
+--------------------+

{%- endcapture -%}

{%- capture api_link -%}
[CamemBertForQuestionAnswering](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/CamemBertForQuestionAnswering)
{%- endcapture -%}

{%- capture python_api_link -%}
[CamemBertForQuestionAnswering](/api/python/reference/autosummary/sparknlp/annotator/classifier_dl/camembert_for_question_answering/index.html#sparknlp.annotator.classifier_dl.camembert_for_question_answering.CamemBertForQuestionAnswering)
{%- endcapture -%}

{%- capture source_link -%}
[CamemBertForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/CamemBertForQuestionAnswering.scala)
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