{%- capture title -%}
DocumentLogRegClassifier
{%- endcapture -%}

{%- capture description -%}
Trains a model to classify documents with a Logarithmic Regression algorithm. Training data requires columns for
text and their label. The result is a trained GenericClassifierModel.
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols("token") \
    .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner()\
    .setInputCols("normalized")\
    .setOutputCol("cleanTokens")\
    .setCaseSensitive(False)

stemmer = Stemmer() \
    .setInputCols("cleanTokens") \
    .setOutputCol("stem")

gen_clf = DocumentLogRegClassifierApproach() \
    .setLabelColumn("category") \
    .setInputCols("stem") \
    .setOutputCol("prediction") 

pipeline = Pipeline().setStages([
    document_assembler,
    tokenizer,
    normalizer,
    stopwords_cleaner,
    stemmer,
    logreg
])

clf_model = pipeline.fit(data)

{%- endcapture -%}

{%- capture approach_scala_example -%}
val document_assembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val normalizer = new Normalizer()
  .setInputCols("token")
  .setOutputCol("normalized")

val stopwords_cleaner = new StopWordsCleaner()
  .setInputCols("normalized")
  .setOutputCol("cleanTokens")
  .setCaseSensitive(false)

val stemmer = new Stemmer()
  .setInputCols("cleanTokens")
  .setOutputCol("stem")

val logreg = new DocumentLogRegClassifierApproach()
  .setInputCols("stem")
  .setLabelCol("category")
  .setOutputCol("prediction")


val pipeline = new Pipeline().setStages(Array(
  document_assembler,
  tokenizer,
  normalizer,
  stopwords_cleaner,
  stemmer,
  logreg))


val clf_model = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture api_link -%}
[DocumentLogRegClassifierApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/classification/DocumentLogRegClassifierApproach.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[DocumentLogRegClassifierApproach](https://nlp.johnsnowlabs.comlicensed/api/python/reference/autosummary/sparknlp_jsl.annotator.DocumentLogRegClassifierApproach.html)

{%- endcapture -%}


{% include templates/licensed_training_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
%}
