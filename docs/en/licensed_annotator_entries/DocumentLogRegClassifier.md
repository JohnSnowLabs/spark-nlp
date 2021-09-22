{%- capture title -%}
DocumentLogRegClassifier
{%- endcapture -%}

{%- capture model_description -%}
Classifies documents with a Logarithmic Regression algorithm.
Currently there are no pretrained models available.
Please see DocumentLogRegClassifierApproach to train your own model.

Please check out the
[Models Hub](https://nlp.johnsnowlabs.com/models) for available models in the future.
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture model_api_link -%}
[DocumentLogRegClassifierModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/classification/DocumentLogRegClassifierModel)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a model to classify documents with a Logarithmic Regression algorithm. Training data requires columns for
text and their label. The result is a trained DocumentLogRegClassifierModel.
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Define pipeline stages to prepare the data
document_assembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")

tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")

normalizer = Normalizer() \
  .setInputCols(["token"]) \
  .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner() \
  .setInputCols(["normalized"]) \
  .setOutputCol("cleanTokens") \
  .setCaseSensitive(False)

stemmer = Stemmer() \
  .setInputCols(["cleanTokens"]) \
  .setOutputCol("stem")

# Define the document classifier and fit training data to it
logreg = DocumentLogRegClassifierApproach() \
  .setInputCols(["stem"]) \
  .setLabelCol("category") \
  .setOutputCol("prediction")

pipeline = Pipeline(stages=[
  document_assembler,
  tokenizer,
  normalizer,
  stopwords_cleaner,
  stemmer,
  logreg
])

model = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture approach_scala_example -%}
// Define pipeline stages to prepare the data
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

// Define the document classifier and fit training data to it
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
  logreg
))

val model = pipeline.fit(trainingData)

{%- endcapture -%}

{%- capture approach_api_link -%}
[DocumentLogRegClassifierApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/classification/DocumentLogRegClassifierApproach)
{%- endcapture -%}


{% include templates/licensed_approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_api_link=approach_api_link
%}
