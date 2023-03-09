#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Package that contains classes for integration with Comet."""

try:
    import comet_ml
except AttributeError:
    # Python 3.6
    comet_ml = None
except ModuleNotFoundError:
    # Python 3.7+
    comet_ml = None

import threading
import time
import os


class CometLogger:
    """Logger class for Comet integration

    `Comet <https://www.comet.ml/>`__ is a meta machine learning platform
    designed to help AI practitioners and teams build reliable machine learning
    models for real-world applications by streamlining the machine learning
    model lifecycle. By leveraging Comet, users can track, compare, explain and
    reproduce their machine learning experiments.

    To log a Spark NLP annotator, it will need an "outputLogPath" parameter, as the
    CometLogger reads the log file generated during the training process.

    For more examples see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/logging/Comet_SparkNLP_Integration.ipynb>`__.

    Parameters
    ----------
    workspace : str, optional
        Name of the workspace in Comet, by default None
    project_name : str, optional
        Name of the project in Comet, by default None
    comet_mode : str, optional
        Mode of logging, by default None. If set to "offline" then offline mode
        will be used, otherwise online.
    experiment_id : str, optional
        Id of the experiment, if it is reused, by default None
    tags : List[str], optional
        List of tags for the experiment, by default None

    Attributes
    ----------
    experiment : comet_ml.Experiment
        Object representing the Comet experiment

    Raises
    ------
    ImportError
        If the package comet-ml is not installed

    Examples
    --------
    Metrics while training an annotator can be logged with for example:

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.logging.comet import CometLogger
    >>> spark = sparknlp.start()

    To run an online experiment, the logger is defined like so.

    >>> OUTPUT_LOG_PATH = "./run"
    >>> logger = CometLogger()

    Then the experiment can start like so

    >>> document = DocumentAssembler() \\
    ...     .setInputCol("text")\\
    ...     .setOutputCol("document")
    >>> embds = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence_embeddings")
    >>> multiClassifier = MultiClassifierDLApproach() \\
    ...     .setInputCols("sentence_embeddings") \\
    ...     .setOutputCol("category") \\
    ...     .setLabelColumn("labels") \\
    ...     .setBatchSize(128) \\
    ...     .setLr(1e-3) \\
    ...     .setThreshold(0.5) \\
    ...     .setShufflePerEpoch(False) \\
    ...     .setEnableOutputLogs(True) \\
    ...     .setOutputLogsPath(OUTPUT_LOG_PATH) \\
    ...     .setMaxEpochs(1)
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
    """

    def __init__(
        self,
        workspace=None,
        project_name=None,
        comet_mode=None,
        experiment_id=None,
        tags=None,
        **experiment_kwargs,
    ):
        if comet_ml is None:
            raise ImportError(
                "`comet_ml` is not installed. Please install it with `pip install comet-ml`."
            )

        self.comet_mode = comet_mode
        self.workspace = workspace
        self.project_name = project_name
        self.experiment_id = experiment_id
        self.experiment_kwargs = experiment_kwargs

        self.experiment = self._get_experiment(
            self.comet_mode,
            self.workspace,
            self.project_name,
            self.experiment_id,
            **self.experiment_kwargs,
        )
        self.experiment.log_other("Created from", "SparkNLP")
        if tags is not None:
            self.experiment.add_tags(tags)

        self._watch_file = False
        self._monitor_thread_timeout = 5
        self.thread = None

    def _get_experiment(
        self,
        mode,
        workspace=None,
        project_name=None,
        experiment_id=None,
        **experiment_kwargs,
    ):
        if mode == "offline":
            if experiment_id is not None:
                return comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id,
                    workspace=workspace,
                    project_name=project_name,
                    **experiment_kwargs,
                )

            return comet_ml.OfflineExperiment(
                workspace=workspace,
                project_name=project_name,
                **experiment_kwargs,
            )

        else:
            if experiment_id is not None:
                return comet_ml.ExistingExperiment(
                    previous_experiment=experiment_id,
                    workspace=workspace,
                    project_name=project_name,
                    **experiment_kwargs,
                )

            return comet_ml.Experiment(
                workspace=workspace,
                project_name=project_name,
                **experiment_kwargs,
            )

    def log_pipeline_parameters(self, pipeline, stages=None):
        """Iterates over the different stages in a pyspark PipelineModel object
        and logs the parameters to Comet.

        Parameters
        ----------
        pipeline : pyspark.ml.PipelineModel
            PipelineModel object
        stages : List[str], optional
            Names of the stages of the pipeline to include, by default None (logs all)

        Examples
        --------
        The pipeline model contains the annotators of Spark NLP, that were
        fitted to a dataframe.

        >>> logger.log_pipeline_parameters(pipeline_model)
        """
        self.experiment.log_other("pipeline_uid", pipeline.uid)
        if stages is None:
            stages = [s.name for s in pipeline.stages]

        for stage in pipeline.stages:
            if stage.name not in stages:
                continue

            params = stage.extractParamMap()
            for param, param_value in params.items():
                self.experiment.log_parameter(f"{stage.name}-{param.name}", param_value)

    def log_visualization(self, html, name="viz.html"):
        """Uploads a NER visualization from Spark NLP Display to comet.

        Parameters
        ----------
        html : str
            HTML of the spark NLP Display visualization
        name : str, optional
            Name for the visualization in comet, by default "viz.html"

        Examples
        --------
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
        """
        self.log_asset_data(html, name)

    def log_metrics(self, metrics, step=None, epoch=None, prefix=None):
        """Submits logs of an evaluation metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary with key value pairs corresponding to the measured metric
            and its value
        step : int, optional
            Used to associate a specific step, by default None
        epoch : int, optional
            Used to associate a specific epoch, by default None
        prefix : str, optional
            Name prefix for this metric, by default None. This can be used to
            identify for example different features by name.

        Examples
        --------
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
        """
        self.experiment.log_metrics(metrics, step=step, epoch=epoch, prefix=prefix)

    def log_parameters(self, parameters, step=None):
        """Logs a dictionary (or dictionary-like object) of multiple parameters.

        Parameters
        ----------
        parameters : dict
            Parameters in a key : value form
        step : int, optional
            Used to associate a specific step, by default None, by default None
        """
        self.experiment.log_parameters(parameters, step=step)

    def log_completed_run(self, log_file_path):
        """Submit logs of training metrics after a run has completed.

        Parameters
        ----------
        log_file_path : str
            Path to log file containing training metrics
        """
        with open(log_file_path, "r") as f:
            stats = f.read().splitlines()

        self._parse_log_entry(stats)
        self.experiment.log_other("log_file_path", log_file_path)

    def log_asset(self, asset_path, metadata=None, step=None):
        """Uploads an asset to comet.

        Parameters
        ----------
        asset_path : str
            Path to the asset
        metadata : str, optional
            Some additional data to attach to the the audio asset. Must be a
            JSON-encodable dict, by default None
        step : int, optional
            Used to associate a specific step, by default None, by default None
        """
        self.experiment.log_asset(asset_path, metadata=metadata, step=step)

    def log_asset_data(self, asset, name, overwrite=False, metadata=None, step=None):
        """Uploads the data given to comet (str, binary, or JSON).

        Parameters
        ----------
        asset : str or bytes or dict
            Data to be saved as asset
        name : str
            A custom file name to be displayed
        overwrite : bool, optional
            If True will overwrite all existing assets with the same name, by
            default False
        metadata : dict, optional
            Some additional data to attach to the the asset data.
            Must be a JSON-encodable dict, by default None
        step : int, optional
            Used to associate a specific step, by default None, by default None
        """
        self.experiment.log_asset_data(
            asset, name, overwrite=overwrite, metadata=metadata, step=step
        )

    def monitor(self, logdir, model, interval=10):
        """Monitors the training of the model and submits logs to comet, given
        by an interval.

        To log a Spark NLP annotator, it will need an "outputLogPath" parameter, as the
        CometLogger reads the log file generated during the training process.

        If you are not able to monitor the live training, you can still log the training
        at the end with :meth:`.log_completed_run`.

        Parameters
        ----------
        logdir : str
            Path to the output of the logs
        model : AnnotatorApproach
            Annotator to monitor
        interval : int, optional
            Interval for refreshing, by default 10
        """
        self._watch_file = True
        self.experiment.log_other("model_uid", model.uid)
        self.thread = threading.Thread(
            target=self._monitor_log_file,
            args=(
                os.path.join(logdir, f"{model.uid}.log"),
                interval,
            ),
        )
        self.thread.start()

    def _file_watcher(self, filename, interval):
        """Generator that yields lines from the model log file.

        Parameters
        ----------
        filename : str
            Path to model log file
        interval : int
            Time (seconds) to wait in between checking for file updates

        Yields
        ------
        str
            A single line from the file
        """
        fp = open(filename)

        line = ""
        while self._watch_file:
            partial_line = fp.readline()
            if len(partial_line) != 0:
                line += partial_line
                if line.endswith("\n"):
                    yield line
                    line = ""
            else:
                time.sleep(interval)

        fp.close()

    def _monitor_log_file(self, filename, interval):
        # Wait for file to be created:
        while not os.path.exists(filename) and self._watch_file:
            time.sleep(interval)

        watcher = self._file_watcher(filename, interval)
        for line in watcher:
            lines = line.split("\n")
            self._parse_log_entry(lines)

    def _convert_log_entry_to_dict(self, log_entries):
        output_dict = {}
        for entry in log_entries:
            key, value = entry.strip(" ").split(":")
            output_dict[key] = float(value)

        return output_dict

    def _parse_run_metrics(self, parts):
        epoch_str, ratio = parts[0].split(" ", 1)
        epoch, total = ratio.split("/", 1)

        metrics = parts[2:]
        formatted_metrics = self._convert_log_entry_to_dict(metrics)

        return formatted_metrics, epoch

    def _parse_run_parameters(self, parts):
        parameters = parts[2:]
        formatted_parameters = self._convert_log_entry_to_dict(parameters)
        return formatted_parameters

    def _parse_log_entry(self, lines):
        for line in lines:
            parts = line.split("-")
            if line.startswith("Training started"):
                parameters = self._parse_run_parameters(parts)
                self.log_parameters(parameters)

            elif line.startswith("Epoch"):
                metrics, epoch = self._parse_run_metrics(parts)
                self.log_metrics(metrics, step=int(epoch), epoch=int(epoch))

    def end(self):
        """Ends the experiment and the logger. Submits all outstanding logs to
        comet.
        """
        self._watch_file = False
        self.experiment.end()
        if self.thread:
            self.thread.join(timeout=self._monitor_thread_timeout)
