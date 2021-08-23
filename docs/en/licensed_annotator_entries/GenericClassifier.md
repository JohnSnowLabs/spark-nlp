{%- capture title -%}
GenericClassifier
{%- endcapture -%}

{%- capture model_description -%}
Creates a generic single-label classifier which uses pre-generated Tensorflow graphs.
The model operates on FEATURE_VECTOR annotations which can be produced using FeatureAssembler.
Requires the FeaturesAssembler to create the input.
{%- endcapture -%}

{%- capture model_input_anno -%}
FEATURE_VECTOR
{%- endcapture -%}

{%- capture model_output_anno -%}
CATEGORY
{%- endcapture -%}

{%- capture model_api_link -%}
[GenericClassifierModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/generic_classifier/GenericClassifierModel)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a TensorFlow model for generic classification of feature vectors. It takes FEATURE_VECTOR annotations from
`FeaturesAssembler` as input, classifies them and outputs CATEGORY annotations.
Please see the Parameters section for required training parameters.

For a more extensive example please see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/8.Generic_Classifier.ipynb).
{%- endcapture -%}

{%- capture approach_input_anno -%}
FEATURE_VECTOR
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
features_asm = FeaturesAssembler() \
    .setInputCols(["feature_1", "feature_2", "...", "feature_n"]) \
    .setOutputCol("features")

gen_clf = GenericClassifierApproach() \
    .setLabelColumn("target") \
    .setInputCols(["features"]) \
    .setOutputCol("prediction") \
    .setModelFile("/path/to/graph_file.pb") \
    .setEpochsNumber(50) \
    .setBatchSize(100) \
    .setFeatureScaling("zscore") \
    .setlearningRate(0.001) \
    .setFixImbalance(True) \
    .setOutputLogsPath("logs") \
    .setValidationSplit(0.2) # keep 20% of the data for validation purposes

pipeline = Pipeline().setStages([
    features_asm,
    gen_clf
])

clf_model = pipeline.fit(data)

{%- endcapture -%}

{%- capture approach_scala_example -%}
val features_asm = new FeaturesAssembler()
  .setInputCols(Array("feature_1", "feature_2", "...", "feature_n"))
  .setOutputCol("features")

val gen_clf = new GenericClassifierApproach()
  .setLabelColumn("target")
  .setInputCols("features")
  .setOutputCol("prediction")
  .setModelFile("/path/to/graph_file.pb")
  .setEpochsNumber(50)
  .setBatchSize(100)
  .setFeatureScaling("zscore")
  .setlearningRate(0.001f)
  .setFixImbalance(true)
  .setOutputLogsPath("logs")
  .setValidationSplit(0.2f) // keep 20% of the data for validation purposes

val pipeline = new Pipeline().setStages(Array(
  features_asm,
  gen_clf
))

val clf_model = pipeline.fit(data)

{%- endcapture -%}

{%- capture approach_api_link -%}
[GenericClassifierApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/generic_classifier/GenericClassifierApproach)
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
