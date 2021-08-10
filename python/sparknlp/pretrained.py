#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Contains classes for the :class:`PretrainedPipeline` and downloading Pretrained Models.
"""

import sparknlp.internal as _internal
import threading
import time
from pyspark.sql import DataFrame
from sparknlp.annotator import *
from sparknlp.base import LightPipeline
from pyspark.ml import PipelineModel
from py4j.protocol import Py4JJavaError


def printProgress(stop):
    states = [' | ', ' / ', ' — ', ' \\ ']
    nextc = 0
    while True:
        sys.stdout.write('\r[{}]'.format(states[nextc]))
        sys.stdout.flush()
        time.sleep(2.5)
        nextc = nextc + 1 if nextc < 3 else 0
        if stop():
            sys.stdout.write('\r[{}]'.format('OK!'))
            sys.stdout.flush()
            break

    sys.stdout.write('\n')
    return


class ResourceDownloader(object):

    @staticmethod
    def downloadModel(reader, name, language, remote_loc=None, j_dwn='PythonResourceDownloader'):
        print(name + " download started this may take some time.")
        file_size = _internal._GetResourceSize(name, language, remote_loc).apply()
        if file_size == "-1":
            print("Can not find the model to download please check the name!")
        else:
            print("Approximate size to download " + file_size)
            stop_threads = False
            t1 = threading.Thread(target=printProgress, args=(lambda: stop_threads,))
            t1.start()
            try:
                j_obj = _internal._DownloadModel(reader.name, name, language, remote_loc, j_dwn).apply()
            except Py4JJavaError as e:
                sys.stdout.write("\n" + str(e))
                raise e
            finally:
                stop_threads = True
                t1.join()

            return reader(classname=None, java_model=j_obj)

    @staticmethod
    def downloadPipeline(name, language, remote_loc=None):
        print(name + " download started this may take some time.")
        file_size = _internal._GetResourceSize(name, language, remote_loc).apply()
        if file_size == "-1":
            print("Can not find the model to download please check the name!")
        else:
            print("Approx size to download " + file_size)
            stop_threads = False
            t1 = threading.Thread(target=printProgress, args=(lambda: stop_threads,))
            t1.start()
            try:
                j_obj = _internal._DownloadPipeline(name, language, remote_loc).apply()
                jmodel = PipelineModel._from_java(j_obj)
            finally:
                stop_threads = True
                t1.join()

            return jmodel

    @staticmethod
    def clearCache(name, language, remote_loc=None):
        _internal._ClearCache(name, language, remote_loc).apply()

    @staticmethod
    def showPublicModels():
        _internal._ShowPublicModels().apply()

    @staticmethod
    def showPublicPipelines():
        _internal._ShowPublicPipelines().apply()


    @staticmethod
    def showUnCategorizedResources():
        _internal._ShowUnCategorizedResources().apply()


class PretrainedPipeline:
    """Loads a Represents a fully constructed and trained Spark NLP pipeline,
    ready to be used.

    This way, a whole pipeline can be defined in 1 line. Additionally, the
    :class:`.LightPipeline` version of the model can be retrieved with member
    :attr:`.light_model`.

    For more extended examples see the `Pipelines page
    <https://nlp.johnsnowlabs.com/docs/en/pipelines>`_ and our `Github Model
    Repository <https://github.com/JohnSnowLabs/spark-nlp-models>`_  for
    available pipeline models.

    Parameters
    ----------
    name : str
        Name of the PretrainedPipeline. These can be gathered from the Pipelines
        Page.
    lang : str, optional
        Langauge of the model, by default 'en'
    remote_loc : str, optional
        Link to the remote location of the model (if it was already downloaded),
        by default None
    parse_embeddings : bool, optional
        Whether to parse embeddings, by default False
    disk_location : str , optional
        Path to locally stored PretrainedPipeline, by default None
    """

    def __init__(self, name, lang='en', remote_loc=None, parse_embeddings=False, disk_location=None):
        if not disk_location:
            self.model = ResourceDownloader().downloadPipeline(name, lang, remote_loc)
        else:
            self.model = PipelineModel.load(disk_location)
        self.light_model = LightPipeline(self.model, parse_embeddings)

    @staticmethod
    def from_disk(path, parse_embeddings=False):
        return PretrainedPipeline(None, None, None, parse_embeddings, path)

    def annotate(self, target, column=None):
        """Annotates the data provided, extracting the results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated

        Returns
        -------
        List[dict] or dict
            The result of the annotation

        Examples
        --------
        >>> from sparknlp.pretrained import PretrainedPipeline
        >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
        >>> result = explain_document_pipeline.annotate('U.N. official Ekeus heads for Baghdad.')
        >>> result.keys()
        dict_keys(['entities', 'stem', 'checked', 'lemma', 'document', 'pos', 'token', 'ner', 'embeddings', 'sentence'])
        >>> result["ner"]
        ['B-ORG', 'O', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'O']
        """
        if type(target) is DataFrame:
            if not column:
                raise Exception("annotate() column arg needed when targeting a DataFrame")
            return self.model.transform(target.withColumnRenamed(column, "text"))
        elif type(target) is list or type(target) is str:
            pipeline = self.light_model
            return pipeline.annotate(target)
        else:
            raise Exception("target must be either a spark DataFrame, a list of strings or a string")

    def fullAnnotate(self, target, column=None):
        """Annotates the data provided into `Annotation` type results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated

        Returns
        -------
        List[Annotation]
            The result of the annotation

        Examples
        --------
        >>> from sparknlp.pretrained import PretrainedPipeline
        >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
        >>> result = explain_document_pipeline.fullAnnotate('U.N. official Ekeus heads for Baghdad.')
        >>> result[0].keys()
        dict_keys(['entities', 'stem', 'checked', 'lemma', 'document', 'pos', 'token', 'ner', 'embeddings', 'sentence'])
        >>> result[0]["ner"]
        [Annotation(named_entity, 0, 2, B-ORG, {'word': 'U.N'}),
        Annotation(named_entity, 3, 3, O, {'word': '.'}),
        Annotation(named_entity, 5, 12, O, {'word': 'official'}),
        Annotation(named_entity, 14, 18, B-PER, {'word': 'Ekeus'}),
        Annotation(named_entity, 20, 24, O, {'word': 'heads'}),
        Annotation(named_entity, 26, 28, O, {'word': 'for'}),
        Annotation(named_entity, 30, 36, B-LOC, {'word': 'Baghdad'}),
        Annotation(named_entity, 37, 37, O, {'word': '.'})]
        """
        if type(target) is DataFrame:
            if not column:
                raise Exception("annotate() column arg needed when targeting a DataFrame")
            return self.model.transform(target.withColumnRenamed(column, "text"))
        elif type(target) is list or type(target) is str:
            pipeline = self.light_model
            return pipeline.fullAnnotate(target)
        else:
            raise Exception("target must be either a spark DataFrame, a list of strings or a string")

    def transform(self, data):
        """Transforms the input dataset with Spark.

        Parameters
        ----------
        data : :class:`pyspark.sql.DataFrame`
            input dataset

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            transformed dataset
        """
        return self.model.transform(data)
