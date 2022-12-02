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
"""Contains classes for Annotator properties."""

from pyspark.ml.param import Param, Params, TypeConverters


class HasBatchedAnnotate:
    batchSize = Param(Params._dummy(), "batchSize", "Size of every batch", TypeConverters.toInt)

    def setBatchSize(self, v):
        """Sets batch size.

        Parameters
        ----------
        v : int
            Batch size
        """
        return self._set(batchSize=v)

    def getBatchSize(self):
        """Gets current batch size.

        Returns
        -------
        int
            Current batch size
        """
        return self.getOrDefault("batchSize")


class HasCaseSensitiveProperties:
    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in tokens for embeddings matching",
                          typeConverter=TypeConverters.toBoolean)

    def setCaseSensitive(self, value):
        """Sets whether to ignore case in tokens for embeddings matching.

        Parameters
        ----------
        value : bool
            Whether to ignore case in tokens for embeddings matching
        """
        return self._set(caseSensitive=value)

    def getCaseSensitive(self):
        """Gets whether to ignore case in tokens for embeddings matching.

        Returns
        -------
        bool
            Whether to ignore case in tokens for embeddings matching
        """
        return self.getOrDefault(self.caseSensitive)


class HasClassifierActivationProperties:
    activation = Param(Params._dummy(),
                       "activation",
                       "Whether to calculate logits via Softmax or Sigmoid. Default is Softmax",
                       typeConverter=TypeConverters.toString)

    def setActivation(self, value):
        """Sets whether to calculate logits via Softmax or Sigmoid. Default is Softmax

        Parameters
        ----------
        value : str
            Whether to calculate logits via Softmax or Sigmoid. Default is Softmax
        """
        return self._set(activation=value)

    def getActivation(self):
        """Gets whether to calculate logits via Softmax or Sigmoid. Default is Softmax

        Returns
        -------
        str
            Whether to calculate logits via Softmax or Sigmoid. Default is Softmax
        """
        return self.getOrDefault(self.activation)


class HasEmbeddingsProperties(Params):
    dimension = Param(Params._dummy(),
                      "dimension",
                      "Number of embedding dimensions",
                      typeConverter=TypeConverters.toInt)

    def setDimension(self, value):
        """Sets embeddings dimension.

        Parameters
        ----------
        value : int
            Embeddings dimension
        """
        return self._set(dimension=value)

    def getDimension(self):
        """Gets embeddings dimension."""
        return self.getOrDefault(self.dimension)


class HasEnableCachingProperties:
    enableCaching = Param(Params._dummy(),
                          "enableCaching",
                          "Whether to enable caching DataFrames or RDDs during the training",
                          typeConverter=TypeConverters.toBoolean)

    def setEnableCaching(self, value):
        """Sets whether to enable caching DataFrames or RDDs during the training

        Parameters
        ----------
        value : bool
            Whether to enable caching DataFrames or RDDs during the training
        """
        return self._set(enableCaching=value)

    def getEnableCaching(self):
        """Gets whether to enable caching DataFrames or RDDs during the training

        Returns
        -------
        bool
            Whether to enable caching DataFrames or RDDs during the training
        """
        return self.getOrDefault(self.enableCaching)


class HasBatchedAnnotateImage:
    batchSize = Param(Params._dummy(), "batchSize", "Size of every batch", TypeConverters.toInt)

    def setBatchSize(self, v):
        """Sets batch size.

        Parameters
        ----------
        v : int
            Batch size
        """
        return self._set(batchSize=v)

    def getBatchSize(self):
        """Gets current batch size.

        Returns
        -------
        int
            Current batch size
        """
        return self.getOrDefault("batchSize")


class HasImageFeatureProperties:
    doResize = Param(Params._dummy(), "doResize", "Whether to resize the input to a certain size",
                     TypeConverters.toBoolean)

    doNormalize = Param(Params._dummy(), "doNormalize",
                        "Whether to normalize the input with mean and standard deviation",
                        TypeConverters.toBoolean)

    featureExtractorType = Param(Params._dummy(), "featureExtractorType",
                                 "Name of model's architecture for feature extraction",
                                 TypeConverters.toString)

    imageMean = Param(Params._dummy(), "imageMean",
                      "The sequence of means for each channel, to be used when normalizing images",
                      TypeConverters.toListFloat)

    imageStd = Param(Params._dummy(), "imageStd",
                     "The sequence of standard deviations for each channel, to be used when normalizing images",
                     TypeConverters.toListFloat)

    resample = Param(Params._dummy(), "resample",
                     "An optional resampling filter. This can be one of PIL.Image.NEAREST, PIL.Image.BOX, "
                     "PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC or PIL.Image.LANCZOS. Only has an "
                     "effect if do_resize is set to True",
                     TypeConverters.toInt)

    size = Param(Params._dummy(), "size",
                 "Resize the input to the given size. If a tuple is provided, it should be (width, height). If only "
                 "an integer is provided, then the input will be resized to (size, size). Only has an effect if "
                 "do_resize is set to True.",
                 TypeConverters.toInt)

    def setDoResize(self, value):
        """

        Parameters
        ----------
        value : Boolean
            Whether to resize the input to a certain size
        """
        return self._set(doResize=value)

    def setDoNormalize(self, value):
        """

        Parameters
        ----------
        value : Boolean
            Whether to normalize the input with mean and standard deviation
        """
        return self._set(doNormalize=value)

    def setFeatureExtractorType(self, value):
        """

        Parameters
        ----------
        value : str
            Name of model's architecture for feature extraction
        """
        return self._set(featureExtractorType=value)

    def setImageStd(self, value):
        """

        Parameters
        ----------
        value : List[float]
            The sequence of standard deviations for each channel, to be used when normalizing images
        """
        return self._set(imageStd=value)

    def setImageMean(self, value):
        """

        Parameters
        ----------
        value : List[float]
            The sequence of means for each channel, to be used when normalizing images
        """
        return self._set(imageMean=value)

    def setResample(self, value):
        """

        Parameters
        ----------
        value : int
            An optional resampling filter. This can be one of PIL.Image.NEAREST,
        PIL.Image.BOX, PIL.Image.BILINEAR PIL.Image.HAMMING, PIL.Image.BICUBIC or PIL.Image.LANCZOS. Only has an
        effect if do_resize is set to True
        """
        return self._set(resample=value)

    def setSize(self, value):
        """

        Parameters
        ----------
        value : int
            Resize the input to the given size. If a tuple is provided, it should be (width, height).
        """
        return self._set(size=value)


class HasBatchedAnnotateAudio:
    batchSize = Param(Params._dummy(), "batchSize", "Size of every batch", TypeConverters.toInt)

    def setBatchSize(self, v):
        """Sets batch size.

        Parameters
        ----------
        v : int
            Batch size
        """
        return self._set(batchSize=v)

    def getBatchSize(self):
        """Gets current batch size.

        Returns
        -------
        int
            Current batch size
        """
        return self.getOrDefault("batchSize")


class HasAudioFeatureProperties:
    doNormalize = Param(Params._dummy(), "doNormalize",
                        "Whether to normalize the input",
                        TypeConverters.toBoolean)

    returnAttentionMask = Param(Params._dummy(), "returnAttentionMask", "",
                                TypeConverters.toBoolean)

    paddingSide = Param(Params._dummy(), "paddingSide",
                        "",
                        TypeConverters.toString)

    featureSize = Param(Params._dummy(), "featureSize",
                        "",
                        TypeConverters.toInt)

    samplingRate = Param(Params._dummy(), "samplingRate",
                         "",
                         TypeConverters.toInt)

    paddingValue = Param(Params._dummy(), "paddingValue",
                         "",
                         TypeConverters.toFloat)

    def setDoNormalize(self, value):
        """

        Parameters
        ----------
        value : Boolean
            Whether to normalize the input with mean and standard deviation
        """
        return self._set(doNormalize=value)

    def setReturnAttentionMask(self, value):
        """

        Parameters
        ----------
        value : boolean
        """
        return self._set(returnAttentionMask=value)

    def setPaddingSide(self, value):
        """

        Parameters
        ----------
        value : str

        """
        return self._set(paddingSide=value)

    def setFeatureSize(self, value):
        """

        Parameters
        ----------
        value : int

        """
        return self._set(featureSize=value)

    def setSamplingRate(self, value):
        """

        Parameters
        ----------
        value : Int
        """
        return self._set(samplingRate=value)

    def setPaddingValue(self, value):
        """

        Parameters
        ----------
        value : float
        """
        return self._set(paddingValue=value)


class HasEngine:
    engine = Param(Params._dummy(), "engine",
                   "Deep Learning engine used for this model",
                   typeConverter=TypeConverters.toString)

    def getEngine(self):
        """

        Returns
        -------
        str
           Deep Learning engine used for this model"
        """
        return self.getOrDefault(self.engine)
