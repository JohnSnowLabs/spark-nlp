class AnnotatorProperties(Params):
    inputCols = Param(Params._dummy(),
                      "inputCols",
                      "previous annotations columns, if renamed",
                      typeConverter=TypeConverters.toListString)
    outputCol = Param(Params._dummy(),
                      "outputCol",
                      "output annotation column. can be left default.",
                      typeConverter=TypeConverters.toString)
    lazyAnnotator = Param(Params._dummy(),
                          "lazyAnnotator",
                          "Whether this AnnotatorModel acts as lazy in RecursivePipelines",
                          typeConverter=TypeConverters.toBoolean
                          )

    def setInputCols(self, *value):
        """Sets column names of input annotations.

        Parameters
        ----------
        *value : str
            Input columns for the annotator
        """
        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def getInputCols(self):
        """Gets current column names of input annotations."""
        return self.getOrDefault(self.inputCols)

    def setOutputCol(self, value):
        """Sets output column name of annotations.

        Parameters
        ----------
        value : str
            Name of output column
        """
        return self._set(outputCol=value)

    def getOutputCol(self):
        """Gets output column name of annotations."""
        return self.getOrDefault(self.outputCol)

    def setLazyAnnotator(self, value):
        """Sets whether Annotator should be evaluated lazily in a
        RecursivePipeline.

        Parameters
        ----------
        value : bool
            Whether Annotator should be evaluated lazily in a
            RecursivePipeline
        """
        return self._set(lazyAnnotator=value)

    def getLazyAnnotator(self):
        """Gets whether Annotator should be evaluated lazily in a
        RecursivePipeline.
        """
        return self.getOrDefault(self.lazyAnnotator)

