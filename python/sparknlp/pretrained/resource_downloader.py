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
"""Contains classes for the ResourceDownloader."""

import sys
import threading

from py4j.protocol import Py4JJavaError
from pyspark.ml import PipelineModel

import sparknlp.internal as _internal
from sparknlp.pretrained.utils import printProgress


class ResourceDownloader(object):
    """Downloads and manages resources, pretrained models/pipelines.

    Usually you will not need to use this class directly. It is called by the
    `pretrained()` function of annotators.

    However, you can use this class to list the available pretrained resources.

    Examples
    --------
    If you want to list all NerDLModels for the english language you can run:

    >>> ResourceDownloader.showPublicModels("NerDLModel", "en")
    +-------------+------+---------+
    | Model       | lang | version |
    +-------------+------+---------+
    | onto_100    | en   | 2.1.0   |
    | onto_300    | en   | 2.1.0   |
    | ner_dl_bert | en   | 2.2.0   |
    |  ...        | ...  | ...     |


    Similarly for Pipelines:

    >>> ResourceDownloader.showPublicPipelines("en")
    +------------------+------+---------+
    | Pipeline         | lang | version |
    +------------------+------+---------+
    | dependency_parse | en   | 2.0.2   |
    | check_spelling   | en   | 2.1.0   |
    | match_datetime   | en   | 2.1.0   |
    |  ...             | ...  | ...     |

    """

    @staticmethod
    def downloadModel(reader, name, language, remote_loc=None, j_dwn='PythonResourceDownloader'):
        """Downloads and loads a model with the default downloader. Usually this method
        does not need to be called directly, as it is called by the `pretrained()`
        method of the annotator.

        Parameters
        ----------
        reader : obj
           Class to read the model for
        name : str
            Name of the pretrained model
        language : str
            Language of the model
        remote_loc : str, optional
            Directory of the Spark NLP Folder, by default None
        j_dwn : str, optional
            Which java downloader to use, by default 'PythonResourceDownloader'

        Returns
        -------
        AnnotatorModel
            Loaded pretrained annotator/pipeline
        """
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
    def downloadModelDirectly(name, remote_loc="public/models", unzip=True):
        """Downloads a model directly to the cache folder.
        You can use to copy-paste the s3 URI from the model hub  and download the model.
        For available s3 URI and models, please see the `Models Hub <https://sparknlp.org/models>`__.
        Parameters
        ----------
        name : str
            Name of the model or s3 URI
        remote_loc : str, optional
            Directory of the remote Spark NLP Folder, by default "public/models"
        unzip : Bool, optional
            Used to unzip model, by default 'True'
        """
        _internal._DownloadModelDirectly(name, remote_loc, unzip).apply()


    @staticmethod
    def downloadPipeline(name, language, remote_loc=None):
        """Downloads and loads a pipeline with the default downloader.

        Parameters
        ----------
        name : str
            Name of the pipeline
        language : str
            Language of the pipeline
        remote_loc : str, optional
            Directory of the remote Spark NLP Folder, by default None

        Returns
        -------
        PipelineModel
            The loaded pipeline
        """
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
        """Clears the cache entry of a model.

        Parameters
        ----------
        name : str
            Name of the model
        language : en
            Language of the model
        remote_loc : str, optional
            Directory of the remote Spark NLP Folder, by default None
        """
        _internal._ClearCache(name, language, remote_loc).apply()

    @staticmethod
    def showPublicModels(annotator=None, lang=None, version=None):
        """Prints all pretrained models for a particular annotator model, that are
        compatible with a version of Spark NLP. If any of the optional arguments are not
        set, the filter is not considered.

        Parameters
        ----------
        annotator : str, optional
            Name of the annotator to filer, by default None
        lang : str, optional
            Language of the models to filter, by default None
        version : str, optional
            Version of Spark NLP to filter, by default None
        """
        print(_internal._ShowPublicModels(annotator, lang, version).apply())

    @staticmethod
    def showPublicPipelines(lang=None, version=None):
        """Prints all pretrained models for a particular annotator model, that are
        compatible with a version of Spark NLP. If any of the optional arguments are not
        set, the filter is not considered.

        Parameters
        ----------
        lang : str, optional
            Language of the models to filter, by default None
        version : str, optional
            Version of Spark NLP to filter, by default None
        """
        print(_internal._ShowPublicPipelines(lang, version).apply())

    @staticmethod
    def showUnCategorizedResources():
        """Shows models or pipelines in the metadata which has not been categorized yet.
        """
        print(_internal._ShowUnCategorizedResources().apply())

    @staticmethod
    def showAvailableAnnotators():
        """Shows all available annotators in Spark NLP.
        """
        print(_internal._ShowAvailableAnnotators().apply())

