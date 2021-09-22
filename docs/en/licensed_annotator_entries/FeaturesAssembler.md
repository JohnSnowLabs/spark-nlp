{%- capture title -%}
FeaturesAssembler
{%- endcapture -%}

{%- capture description -%}
The FeaturesAssembler is used to collect features from different columns. It can collect features from single value
columns (anything which can be cast to a float, if casts fails then the value is set to 0), array columns or
SparkNLP annotations (if the annotation is an embedding, it takes the embedding, otherwise tries to cast the
`result` field). The output of the transformer is a `FEATURE_VECTOR` annotation (the numeric vector is in the
`embeddings` field).
{%- endcapture -%}

{%- capture input_anno -%}
NONE
{%- endcapture -%}

{%- capture output_anno -%}
"feature_vector"
{%- endcapture -%}

{%- capture api_link -%}
[FeaturesAssembler](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/FeaturesAssembler)
{%- endcapture -%}

{%- capture python_example -%}
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

{%- capture scala_example -%}
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

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}