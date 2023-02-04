..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

########################################
Comet - A meta machine learning platform
########################################

`Comet <https://www.comet.ml/>`__ is a meta machine learning platform
designed to help AI practitioners and teams build reliable machine learning
models for real-world applications by streamlining the machine learning
model lifecycle. By leveraging Comet, users can track, compare, explain and
reproduce their machine learning experiments.

Comet can easily integrated into the Spark NLP workflow with the a dedicated
logging class :class:`.CometLogger`, to log training and evaluation metrics,
pipeline parameters and NER visualization made with sparknlp-display.

To log a Spark NLP annotator, it will need an "outputLogPath" parameter, as the
CometLogger reads the log file generated during the training process.

For more examples see the `Examples
<https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/logging/Comet_SparkNLP_Integration.ipynb>`__

************
Installation
************

To use the comet logger, you need to set up an account and install comet.

See `Quick Start - Comet.ml <https://www.comet.ml/docs/quick-start/>`__ for more
information.


**************************
Using Comet with Spark NLP
**************************

Comet can be used to monitor the training process of an annotator. Loss,
training accuracy, validation accuracy etc. can be submitted to Comet and will
be available for inspection on the web interface.

The following example shows how to integrate it while training a
:class:`.MultiClassifierDLApproach`.

First we need to do the necessary imports for the pipeline and Comet.

>>> import sparknlp
>>> from sparknlp.base import *
>>> from sparknlp.annotator import *
>>> from sparknlp.logging.comet import CometLogger
>>> spark = sparknlp.start()

The Comet experiment can then be initialized. For this step you will need an API
key. See the Comet quick-start section on how to get one.

To run an online experiment, the logger is defined like so.

>>> OUTPUT_LOG_PATH = "./run"
>>> logger = CometLogger()

We continue by defining the training pipeline.

>>> document = DocumentAssembler() \
...     .setInputCol("text") \
...     .setOutputCol("document")
>>> embds = UniversalSentenceEncoder.pretrained() \
...     .setInputCols("document") \
...     .setOutputCol("sentence_embeddings")
>>> multiClassifier = MultiClassifierDLApproach() \
...     .setInputCols("sentence_embeddings") \
...     .setOutputCol("category") \
...     .setLabelColumn("labels") \
...     .setBatchSize(128) \
...     .setLr(1e-3) \
...     .setThreshold(0.5) \
...     .setShufflePerEpoch(False) \
...     .setEnableOutputLogs(True) \
...     .setOutputLogsPath(OUTPUT_LOG_PATH) \
...     .setMaxEpochs(1)

Comet monitors the output logs of the annotators during the training process.
Before starting the training, the path to the output logs need to be monitored
by the logger.

>>> logger.monitor(logdir=OUTPUT_LOG_PATH, model=multiClassifier)
>>> trainDataset = spark.createDataFrame(
...     [("Nice.", ["positive"]), ("That's bad.", ["negative"])],
...     schema=["text", "labels"],
... )
>>> pipeline = Pipeline(stages=[document, embds, multiClassifier])
>>> pipeline.fit(trainDataset)
>>> logger.end()

If you are using a jupyter notebook, it is possible to display the live web
interface with

>>> logger.experiment.display(tab='charts')

***************************
Logging Pipeline Parameters
***************************

The pipeline model contains the annotators of Spark NLP, that were
fitted to a dataframe.

>>> logger.log_pipeline_parameters(pipeline_model)

**************************
Logging Evaluation Metrics
**************************

In this example, sklearn is used to retrieve the metrics.

>>> from sklearn.preprocessing import MultiLabelBinarizer
>>> from sklearn.metrics import classification_report
>>> prediction = model.transform(testDataset)
>>> preds_df = prediction.select('labels', 'category.result').toPandas()

>>> mlb = MultiLabelBinarizer()
>>> y_true = mlb.fit_transform(preds_df['labels'])
>>> y_pred = mlb.fit_transform(preds_df['result'])
>>> report = classification_report(y_true, y_pred, output_dict=True)

Iterate over the report and log the metrics:

>>> for key, value in report.items():
...     logger.log_metrics(value, prefix=key)
>>> logger.end()

If you are using Spark NLP in a notebook, then you can display the
metrics directly with

>>> logger.experiment.display(tab='metrics')

**********************
Logging Visualizations
**********************

Visualizations from Spark NLP Display can also be submitted to comet.

This example has NER chunks (NER extracted by e.g. :class:`.NerDLModel`
and converted by a :class:`.NerConverter`) extracted in the colum
"ner_chunk".

>>> from sparknlp_display import NerVisualizer
>>> logger = CometLogger()
>>> for idx, result in enumerate(results.collect()):
...     viz = NerVisualizer().display(
...         result=result,
...         label_col='ner_chunk',
...         document_col='document',
...         return_html=True
...     )
...     logger.log_visualization(viz, name=f'viz-{idx}.html')

The visualizations will then be available in comet.

*****************************
Running An Offline Experiment
*****************************

Experiments can also be defined to be run offline:

>>> import comet_ml
>>> OUTPUT_LOG_PATH = "./run"
>>> comet_ml.init(project_name="sparknlp_experiment", offline_directory="/tmp")
>>> logger = CometLogger(comet_mode="offline", offline_directory=OUTPUT_LOG_PATH)

Logs will be saved to the output directory and can be later submitted to
Comet.
