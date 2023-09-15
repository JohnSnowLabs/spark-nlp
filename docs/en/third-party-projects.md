---
layout: docs
header: true
seotitle: Spark NLP
title: Third Party Projects
permalink: /docs/en/third-party-projects
key: docs-third-party-projects
modify_date: "2021-10-25"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

There are third party projects that can integrate with Spark NLP. These packages need to be installed separately to be used.

If you'd like to integrate your application with Spark NLP, please send us a message!

## Logging

### Comet

[Comet](https://www.comet.ml/) is a meta machine learning platform designed to help AI practitioners and teams build reliable machine learning models for real-world applications by streamlining the machine learning model lifecycle. By leveraging Comet, users can track, compare, explain and reproduce their machine learning experiments.

Comet can easily integrated into the Spark NLP workflow with the a dedicated logging class `CometLogger` to log training and evaluation metrics, pipeline parameters and NER visualization made with sparknlp-display.

For more information see the [User Guide](/api/python/third_party/Comet.html) and for more examples see the [Spark NLP Examples](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/logging/Comet_SparkNLP_Integration.ipynb).


| **Python API:** [CometLogger](/api/python/reference/autosummary/sparknlp/logging/comet/index.html#sparknlp.logging.comet.CometLogger) |

<details>

<summary class="button"><b>Show Example</b></summary>

<div class="tabs-box tabs-new" markdown="1">

```python
# Metrics while training an annotator can be logged with for example:

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.logging.comet import CometLogger

spark = sparknlp.start()

OUTPUT_LOG_PATH = "./run"
logger = CometLogger()

document = DocumentAssembler().setInputCol("text").setOutputCol("document")
embds = (
    UniversalSentenceEncoder.pretrained()
    .setInputCols("document")
    .setOutputCol("sentence_embeddings")
)
multiClassifier = (
    MultiClassifierDLApproach()
    .setInputCols("sentence_embeddings")
    .setOutputCol("category")
    .setLabelColumn("labels")
    .setBatchSize(128)
    .setLr(1e-3)
    .setThreshold(0.5)
    .setShufflePerEpoch(False)
    .setEnableOutputLogs(True)
    .setOutputLogsPath(OUTPUT_LOG_PATH)
    .setMaxEpochs(1)
)

logger.monitor(logdir=OUTPUT_LOG_PATH, model=multiClassifier)
trainDataset = spark.createDataFrame(
    [("Nice.", ["positive"]), ("That's bad.", ["negative"])],
    schema=["text", "labels"],
)

pipeline = Pipeline(stages=[document, embds, multiClassifier])
pipeline.fit(trainDataset)
logger.end()

# If you are using a jupyter notebook, it is possible to display the live web
# interface with

logger.experiment.display(tab='charts')
```

</div>

</details>

</div><div class="h3-box" markdown="1">

### MLflow

Spark NLP uses Spark MLlib Pipelines, what are natively supported by MLFlow.
MLFlow is, as stated in their [official webpage](https://mlflow.org/), an open source platform for the machine learning lifecycle, that includes:
* **Mlflow Tracking**: Record and query experiments: code, data, config, and results
* **MLflow Projects**: Package data science code in a format to reproduce runs on any platform
* **MLflow Models**: Deploy machine learning models in diverse serving environments
* **Model Registry**: Store, annotate, discover, and manage models in a central repository

For more information, please see the complete guide at [Experiment Tracking](/docs/en/mlflow).

</div>