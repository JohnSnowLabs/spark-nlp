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
"""Contains classes for MxbaiEmbeddings."""

from sparknlp.common import *


class MxbaiEmbeddings(AnnotatorModel,
					HasEmbeddingsProperties,
					HasCaseSensitiveProperties,
					HasStorageRef,
					HasBatchedAnnotate,
					HasMaxSentenceLengthLimit):
	"""Sentence embeddings using Mxbai Embeddings.


	Pretrained models can be loaded with :meth:`.pretrained` of the companion
	object:

	>>> embeddings = MxbaiEmbeddings.pretrained() \\
	...     .setInputCols(["document"]) \\
	...     .setOutputCol("Mxbai_embeddings")


	The default model is ``"mxbai_large_v1"``, if no name is provided.

	For available pretrained models please see the
	`Models Hub <https://sparknlp.org/models?q=Mxbai>`__.


	====================== ======================
	Input Annotation types Output Annotation type
	====================== ======================
	``DOCUMENT``            ``SENTENCE_EMBEDDINGS``
	====================== ======================

	Parameters
	----------
	batchSize
		Size of every batch , by default 8
	dimension
		Number of embedding dimensions, by default 768
	caseSensitive
		Whether to ignore case in tokens for embeddings matching, by default False
	maxSentenceLength
		Max sentence length to process, by default 512
	configProtoBytes
		ConfigProto from tensorflow, serialized into byte array.



	Examples
	--------
	>>> import sparknlp
	>>> from sparknlp.base import *
	>>> from sparknlp.annotator import *
	>>> from pyspark.ml import Pipeline
	>>> documentAssembler = DocumentAssembler() \\
	...     .setInputCol("text") \\
	...     .setOutputCol("document")
	>>> embeddings = MxbaiEmbeddings.pretrained() \\
	...     .setInputCols(["document"]) \\
	...     .setOutputCol("embeddings")
	>>> embeddingsFinisher = EmbeddingsFinisher() \\
	...     .setInputCols("embeddings") \\
	...     .setOutputCols("finished_embeddings") \\
	...     .setOutputAsVector(True)
	>>> pipeline = Pipeline().setStages([
	...     documentAssembler,
	...     embeddings,
	...     embeddingsFinisher
	... ])
	>>> data = spark.createDataFrame([["hello world", "hello moon"]]).toDF("text")
	>>> result = pipeline.fit(data).transform(data)
	>>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
	+--------------------------------------------------------------------------------+
	|                                                                          result|
	+--------------------------------------------------------------------------------+
	|[0.50387806, 0.5861606, 0.35129607, -0.76046336, -0.32446072, -0.117674336, 0...|
	|[0.6660665, 0.961762, 0.24854276, -0.1018044, -0.6569202, 0.027635604, 0.1915...|
	+--------------------------------------------------------------------------------+
	"""

	name = "MxbaiEmbeddings"

	inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

	outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS
	poolingStrategy = Param(Params._dummy(),
							"poolingStrategy",
							"Pooling strategy to use for sentence embeddings",
							TypeConverters.toString)

	def setPoolingStrategy(self, value):
		"""Pooling strategy to use for sentence embeddings.

		Available pooling strategies for sentence embeddings are:
		  - `"cls"`: leading `[CLS]` token
		  - `"cls_avg"`: leading `[CLS]` token + mean of all other tokens
		  - `"last"`: embeddings of the last token in the sequence
		  - `"avg"`: mean of all tokens
		  - `"max"`: max of all embedding features of the entire token sequence
		  - `"int"`: An integer number, which represents the index of the token to use as the
			embedding

		Parameters
		----------
		value : str
			Pooling strategy to use for sentence embeddings
		"""

		valid_strategies = {"cls", "cls_avg", "last", "avg", "max"}
		if value in valid_strategies or value.isdigit():
			return self._set(poolingStrategy=value)
		else:
			raise ValueError(f"Invalid pooling strategy: {value}. "
							 f"Valid strategies are: {', '.join(self.valid_strategies)} or an integer.")

	@keyword_only
	def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.MxbaiEmbeddings", java_model=None):
		super(MxbaiEmbeddings, self).__init__(
			classname=classname,
			java_model=java_model
		)
		self._setDefault(
			dimension=1024,
			batchSize=8,
			maxSentenceLength=512,
			caseSensitive=False,
			poolingStrategy="cls"
		)

	@staticmethod
	def loadSavedModel(folder, spark_session):
		"""Loads a locally saved model.

		Parameters
		----------
		folder : str
			Folder of the saved model
		spark_session : pyspark.sql.SparkSession
			The current SparkSession

		Returns
		-------
		MxbaiEmbeddings
			The restored model
		"""
		from sparknlp.internal import _MxbaiEmbeddingsLoader
		jModel = _MxbaiEmbeddingsLoader(folder, spark_session._jsparkSession)._java_obj
		return MxbaiEmbeddings(java_model=jModel)

	@staticmethod
	def pretrained(name="mxbai_large_v1", lang="en", remote_loc=None):
		"""Downloads and loads a pretrained model.

		Parameters
		----------
		name : str, optional
			Name of the pretrained model, by default "mxbai_large_v1"
		lang : str, optional
			Language of the pretrained model, by default "en"
		remote_loc : str, optional
			Optional remote address of the resource, by default None. Will use
			Spark NLPs repositories otherwise.

		Returns
		-------
		MxbaiEmbeddings
			The restored model
		"""
		from sparknlp.pretrained import ResourceDownloader
		return ResourceDownloader.downloadModel(MxbaiEmbeddings, name, lang, remote_loc)
