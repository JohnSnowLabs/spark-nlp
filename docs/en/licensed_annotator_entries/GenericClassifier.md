{%- capture title -%}
GenericClassifier
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture model -%}
model
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

{%- capture approach_python_medical -%}
from johnsnowlabs import *
features_asm = medical.FeaturesAssembler() \
    .setInputCols(["feature_1", "feature_2", "...", "feature_n"]) \
    .setOutputCol("features")

gen_clf = medical.GenericClassifierApproach() \
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

{%- capture approach_python_legal -%}
from johnsnowlabs import *
features_asm = medical.FeaturesAssembler() \
    .setInputCols(["feature_1", "feature_2", "...", "feature_n"]) \
    .setOutputCol("features")

gen_clf = legal.GenericClassifierApproach() \
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


{%- capture approach_python_finance -%}
from johnsnowlabs import *
features_asm = medical.FeaturesAssembler() \
    .setInputCols(["feature_1", "feature_2", "...", "feature_n"]) \
    .setOutputCol("features")

gen_clf = finance.GenericClassifierApproach() \
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

{%- capture approach_scala_medical -%}
from johnsnowlabs import * 
val features_asm = new medical.FeaturesAssembler()
  .setInputCols(Array("feature_1", "feature_2", "...", "feature_n"))
  .setOutputCol("features")

val gen_clf = new medical.GenericClassifierApproach()
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


{%- capture approach_scala_legal -%}
from johnsnowlabs import * 
val features_asm = new medical.FeaturesAssembler()
  .setInputCols(Array("feature_1", "feature_2", "...", "feature_n"))
  .setOutputCol("features")

val gen_clf = new legal.GenericClassifierApproach()
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


{%- capture approach_scala_finance -%}
from johnsnowlabs import * 
val features_asm = new medical.FeaturesAssembler()
  .setInputCols(Array("feature_1", "feature_2", "...", "feature_n"))
  .setOutputCol("features")

val gen_clf = new finance.GenericClassifierApproach()
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


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
approach=approach
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_medical=approach_python_medical
approach_python_legal=approach_python_legal
approach_python_finance=approach_python_finance
approach_scala_medical=approach_scala_medical
approach_scala_legal=approach_scala_legal
approach_scala_finance=approach_scala_finance
approach_api_link=approach_api_link
%}
