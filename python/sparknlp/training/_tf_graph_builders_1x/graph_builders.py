import os
import re

from sparknlp.internal import _ResourceHelper_moveFile
from sparknlp.training._tf_graph_builders.ner_dl.create_graph import create_graph


class WrongTFVersion(Exception):
    pass


class TFGraphBuilder:
    """
    Generic class to create the tensorflow graphs for 'ner_dl', 'generic_classifier', 'assertion_dl', 'relation_extraction' annotators in spark-nlp healthcare

    Examples
    --------
    >>> from sparknlp.training.tfgraphs import tf_graph_1x
    >>> tf_graph_1x.get_models()

    """

    def supports_auto_file_name(self):
        return False

    def get_model_filename(self):
        raise Exception("Not implemented.")

    def check_build_params(self):

        build_params = self.get_build_params()
        required_params = self.get_model_build_params()

        for req_param in required_params:
            if req_param not in build_params:
                if required_params[req_param] is None:
                    raise Exception(f"You need to specify a value for {req_param} in the build parameters.")

    def get_build_params(self):
        return self.__build_params

    def get_build_params_with_defaults(self):
        build_params = self.get_build_params()
        req_build_params = self.get_model_build_params()

        for req_param in req_build_params:
            if (req_param not in build_params) and (req_build_params[req_param] is not None):
                build_params[req_param] = req_build_params[req_param]

        return build_params

    def get_build_param(self, build_param):
        build_params = self.get_build_params()

        if build_param in build_params:
            return build_params[build_param]

        required_params = self.get_model_build_params()

        if (build_param in required_params) and (required_params[build_param] is not None):
            return required_params[build_param]

        raise Exception(f"No value for {build_param} found.")

    def get_model_build_params(self):
        return {}

    def get_model_build_param_explanations(self):
        return {}

    def __init__(self, build_params):
        self.__build_params = build_params

class NerTFGraphBuilder(TFGraphBuilder):
    """
    Class to build the the TF graphs for MedicalNerApproach.

    Examples
    --------

    >>> from sparknlp.training.tfgraphs import tf_graph_1x
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> feat_size = 200
    >>> n_classes = 6
    >>> tf_graph_1x.build("ner_dl", build_params={"embeddings_dim": 200, "nchars": 83,"ntags": 12},model_location="./ner_graphs",model_filename="auto")
    >>> nerTagger = NerDLApproach()\
    >>>                     .setInputCols(["sentence", "token", "embeddings"])\
    >>>                     .setLabelColumn("label")\
    >>>                     .setOutputCol("ner")\
    >>>                     .setMaxEpochs(2)\
    >>>                     .setBatchSize(64)\
    >>>                     .setRandomSeed(0)\
    >>>                     .setVerbose(1)\
    >>>                     .setValidationSplit(0.2)\
    >>>                     .setEvaluationLogExtended(True) \
    >>>                     .setEnableOutputLogs(True)\
    >>>                     .setIncludeConfidence(True)\
    >>>                     .setOutputLogsPath('ner_logs')\
    >>>                     .setGraphFolder('medical_ner_graphs')\
    >>>                     .setEnableMemoryOptimizer(True)

    """

    def supports_auto_file_name(self):
        return True

    def get_model_filename(self):
        return "blstm_{}_{}_{}_{}.pb".format(
            self.get_build_param("ntags"),
            self.get_build_param("embeddings_dim"),
            self.get_build_param("lstm_size"),
            self.get_build_param("nchars"),
        )

    def get_model_build_params(self):
        return {
            "ntags": None,
            "embeddings_dim": 200,
            "nchars": 100,
            "lstm_size": 128,
            "gpu_device": 0
        }

    def get_model_build_param_explanations(self):
        return {
            "ntags": "Number of tags.",
            "embeddings_dim": "Embeddings dimension.",
            "nchars": "Number of chars.",
            "gpu_device": "Device for training.",
            "lstm_size": "Number of LSTM units."
        }

    def build(self, model_location, model_filename):

        if re.match(r'(\w+)://.*', model_location):
            tmp_location = "/tmp/nerModel"
            create_graph(
                model_location=tmp_location,
                model_filename=model_filename,
                ntags=self.get_build_param("ntags"),
                embeddings_dim=self.get_build_param("embeddings_dim"),
                nchars=self.get_build_param("nchars"),
                lstm_size=self.get_build_param("lstm_size"),
                gpu_device=self.get_build_param("gpu_device"),
                is_medical=False,
            )

            file_location = os.path.join(tmp_location, model_filename)
            _ResourceHelper_moveFile(file_location, model_location).apply()

        else:
            create_graph(
                model_location=model_location,
                model_filename=model_filename,
                ntags=self.get_build_param("ntags"),
                embeddings_dim=self.get_build_param("embeddings_dim"),
                nchars=self.get_build_param("nchars"),
                lstm_size=self.get_build_param("lstm_size"),
                gpu_device=self.get_build_param("gpu_device"),
                is_medical=False,
            )


class TFGraphBuilderFactory:
    """
    Factory class to create the different tensorflow graphs for ner_dl
    """

    __model_builders = {
        "ner_dl": NerTFGraphBuilder
    }

    @staticmethod
    def get_models():
        """
        Method that return the available tf models in  spark-nlp healthcare

        Examples
        --------
        >>> from sparknlp.training.tfgraphs import tf_graph_1x
        >>> tf_graph_1x.get_models()

        """
        return list(TFGraphBuilderFactory.__model_builders.keys())

    @staticmethod
    def build(model_name, build_params, model_location, model_filename="auto"):
        """
        Method that create the tf graph.

        Parameters
        ----------
        model_name: str
            The name of the tf model that you want to build. Model available: ner_dl
        build_params: dict
            Configuration params to build the tf graph for the specific model.
        model_location: str
            Path where the model will be saved
        model_filename: str
            Name of the .rb file. If you put auto the filename will be generated.

        Examples
        --------
        >>> from sparknlp.training.tfgraphs import tf_graph_1x
        >>> tf_graph_1x.build("ner_dl", build_params={"embeddings_dim": 200, "nchars": 83,"ntags": 12},model_location="./ner_graphs",model_filename="auto")

        """
        try:
            import tensorflow as tf

            if not tf.__version__.startswith("1.15"):
                raise WrongTFVersion()

        except WrongTFVersion:
            print(tf.version)
            raise Exception("Tensorflow v1.15 is required to build model graphs.")

        except ModuleNotFoundError:
            raise Exception("You need to install Tensorflow 1.15 to be able to build model graphs")

        if model_name not in TFGraphBuilderFactory.__model_builders:
            raise Exception(f"Can't build a graph for {model_name}: model not supported.")

        model = TFGraphBuilderFactory.__model_builders[model_name](build_params)
        model.check_build_params()

        if model_filename == "auto":
            if not model.supports_auto_file_name():
                msg = f"""
                    {model_name} doesn't support automatic filename generation, please specify the filename of the
                    output graph
                """.strip()
                raise Exception(msg)
            else:
                model_filename = model.get_model_filename()

        model.build(model_location, model_filename)
        print("{} graph exported to {}/{}".format(model_name, model_location, model_filename))

    @staticmethod
    def print_model_params(model_name):
        """
        Method that return the params allowed for the tf model.This method return the params with the description for every param.

        Parameters
        ----------
        model_name: str
            The name of the tf model name.Model availables ner_dl,generic_classifier,assertion_dl and relation_extraction

        Examples
        --------
        >>> from sparknlp.training.tfgraphs import tf_graph
        >>> tf_graph.print_model_params("ner_dl")

        """
        if model_name not in TFGraphBuilderFactory.get_models():
            raise Exception(f"Model {model_name} not supported.")

        model = TFGraphBuilderFactory.__model_builders[model_name]({})
        model_params = model.get_model_build_params()
        model_params_descr = model.get_model_build_param_explanations()

        print(f"{model_name} parameters.")
        print("{:<20} {:<10} {:<20} {}".format("Parameter", "Required", "Default value", "Description"))
        for param in model_params:
            if type(model_params[param]) in [list, tuple]:
                default_value = "[" + ", ".join(map(str, model_params[param])) + "]"
            else:
                default_value = model_params[param]

            print("{:<20} {:<10} {:<20} {}".format(
                param,
                "yes" if default_value is None else "no",
                default_value if default_value is not None else "-",
                model_params_descr[param] if param in model_params_descr else ""
            ))
