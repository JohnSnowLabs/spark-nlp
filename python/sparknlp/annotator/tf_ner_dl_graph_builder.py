from pyspark.ml import Model, Estimator
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from sparknlp.common import *


class TFNerDLGraphBuilderModel(Model, DefaultParamsWritable, DefaultParamsReadable):
    def _transform(self, dataset):
        return dataset


class TFNerDLGraphBuilder(Estimator, DefaultParamsWritable, DefaultParamsReadable):

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN, AnnotatorType.WORD_EMBEDDINGS]

    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Labels",
                        typeConverter=TypeConverters.toString)

    inputCols = Param(Params._dummy(),
                      "inputCols",
                      "Input columns",
                      typeConverter=TypeConverters.toListString)

    graphFolder = Param(Params._dummy(), "graphFolder", "Folder path that contain external graph files",
                        TypeConverters.toString)

    graphFile = Param(Params._dummy(), "graphFile", "Graph file name. If empty, default name is generated.",
                      TypeConverters.toString)

    hiddenUnitsNumber = Param(Params._dummy(),
                              "hiddenUnitsNumber",
                              "Number of hidden units",
                              typeConverter=TypeConverters.toInt)

    def setHiddenUnitsNumber(self, value):
        """Sets the number of hidden units for AssertionDLApproach and MedicalNerApproach

        Parameters
        ----------
        value : int
            Number of hidden units for AssertionDLApproach and MedicalNerApproach
        """
        return self._set(hiddenUnitsNumber=value)

    def getHiddenUnitsNumber(self):
        """Gets the number of hidden units for AssertionDLApproach and MedicalNerApproach."""
        return self.getOrDefault(self.hiddenUnitsNumber)

    def setLabelColumn(self, value):
        """Sets the name of the column for data labels.

        Parameters
        ----------
        value : str
            Column for data labels
        """
        return self._set(labelColumn=value)

    def getLabelColumn(self):
        """Gets the name of the label column."""
        return self.getOrDefault(self.labelColumn)

    def setInputCols(self, *value):
        """Sets column names of input annotations.

        Parameters
        ----------
        *value : List[str]
            Input columns for the annotator
        """
        if type(value[0]) == str or type(value[0]) == list:
            self.inputColsValidation(value)
            if len(value) == 1 and type(value[0]) == list:
                return self._set(inputCols=value[0])
            else:
                return self._set(inputCols=list(value))
        else:
            raise TypeError("InputCols datatype not supported. It must be either str or list")

    def inputColsValidation(self, value):
        actual_columns = len(value)
        if type(value[0]) == list:
            actual_columns = len(value[0])

        expected_columns = len(self.inputAnnotatorTypes)

        if actual_columns != expected_columns:
            raise TypeError(
                f"setInputCols in {self.uid} expecting {expected_columns} columns. "
                f"Provided column amount: {actual_columns}. "
                f"Which should be columns from the following annotators: {self.inputAnnotatorTypes}")

    def getInputCols(self):
        """Gets current column names of input annotations."""
        return self.getOrDefault(self.inputCols)

    def setGraphFolder(self, value):
        """Sets folder path that contain external graph files.

        Parameters
        ----------
        value : srt
            Folder path that contain external graph files.
        """
        return self._set(graphFolder=value)

    def getGraphFolder(self):
        """Gets the graph folder."""
        return self.getOrDefault(self.graphFolder)

    def setGraphFile(self, value):
        """Sets the graph file name.

        Parameters
        ----------
        value : srt
            Greaph file name. If set to "auto", then the graph builder will use the model specific default graph
            file name.
        """
        return self._set(graphFile=value)

    def getGraphFile(self):
        """Gets the graph file name."""
        return self.getOrDefault(self.graphFile)

    def _fit(self, dataset):
        from ..training.tfgraphs import tf_graph, tf_graph_1x

        build_params = {}

        from sparknlp.internal import _NerDLGraphBuilder

        params_java = _NerDLGraphBuilder(
            dataset,
            self.getInputCols(),
            self.getLabelColumn())._java_obj
        params = list(map(int, params_java.toString().replace("(", "").replace(")", "").split(",")))
        build_params["ntags"] = params[0]
        build_params["embeddings_dim"] = params[1]
        build_params["nchars"] = params[2]
        if self.getHiddenUnitsNumber() is not None:
            build_params["lstm_size"] = self.getHiddenUnitsNumber()

        graph_file = "auto"
        if self.getGraphFile() is not None:
            graph_file = self.getGraphFile()

        graph_folder = ""
        if self.getGraphFolder() is not None:
            graph_folder = self.getGraphFolder()

        print("Ner DL Graph Builder configuration:")
        print("Graph folder: {}".format(graph_folder))
        print("Graph file name: {}".format(graph_file))
        print("Build params: ", end="")
        print(build_params)

        try:
            tf_graph.build("ner_dl", build_params=build_params, model_location=self.getGraphFolder(),
                           model_filename=graph_file)
        except Exception:
            print("Can't build the tensorflow graph with TF 2 graph factory, attempting TF 1.15 factory")
            try:
                tf_graph_1x.build("ner_dl", build_params=build_params, model_location=self.getGraphFolder())
            except Exception:
                raise Exception("The tensorflow graphs can't be build.")

        return TFNerDLGraphBuilderModel()

    def __init__(self):
        super(TFNerDLGraphBuilder, self).__init__()
        self._setDefault(
            labelColumn=None,
            inputCols=None,
            graphFolder=None,
            graphFile=None,
            hiddenUnitsNumber=None
        )
