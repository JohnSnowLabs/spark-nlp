{%- capture title -%}
FeaturesAssembler
{%- endcapture -%}

{%- capture approach -%}
approach
{%- endcapture -%}

{%- capture approach_description -%}
The FeaturesAssembler is used to collect features from different columns. It can collect features from single value
columns (anything which can be cast to a float, if casts fails then the value is set to 0), array columns or
SparkNLP annotations (if the annotation is an embedding, it takes the embedding, otherwise tries to cast the
`result` field). The output of the transformer is a `FEATURE_VECTOR` annotation (the numeric vector is in the
`embeddings` field).
{%- endcapture -%}

{%- capture approach_input_anno -%}
NONE
{%- endcapture -%}

{%- capture approach_output_anno -%}
"feature_vector"
{%- endcapture -%}

{%- capture approach_api_link -%}
[FeaturesAssembler](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/FeaturesAssembler)
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
  .setLearningRate(0.001) \
  .setFixImbalance(True) \
  .setOutputLogsPath("logs") \
  .setValidationSplit(0.2) # keep 20% of the data for validation purposes

pipeline = Pipeline(stages=[
  features_asm,
  gen_clf
])

clf_model = pipeline.fit(data)
{%- endcapture -%}

{%- capture approach_python_legal -%}
from johnsnowlabs import * 
features_asm = legal.FeaturesAssembler() \
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
  .setLearningRate(0.001) \
  .setFixImbalance(True) \
  .setOutputLogsPath("logs") \
  .setValidationSplit(0.2) # keep 20% of the data for validation purposes

pipeline = Pipeline(stages=[
  features_asm,
  gen_clf
])

clf_model = pipeline.fit(data)
{%- endcapture -%}

{%- capture approach_python_finance -%}
from johnsnowlabs import * 
features_asm = finance.FeaturesAssembler() \
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
  .setLearningRate(0.001) \
  .setFixImbalance(True) \
  .setOutputLogsPath("logs") \
  .setValidationSplit(0.2) # keep 20% of the data for validation purposes

pipeline = Pipeline(stages=[
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
val features_asm = new legal.FeaturesAssembler()
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
val features_asm = new finance.FeaturesAssembler()
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

{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
approach=approach
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_medical=approach_python_medical
approach_python_legal=approach_python_legal
approach_python_finance=approach_python_finance
approach_scala_medical=approach_scala_medical
approach_scala_legal=approach_scala_legal
approach_scala_finance=approach_scala_finance
approach_api_link=approach_api_link%}