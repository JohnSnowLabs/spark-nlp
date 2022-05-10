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

"""Contains all the basic components to create a Spark NLP Pipeline.

This module contains basic transformers and extensions to the Spark Pipeline
interface. These are the :class:`LightPipeline` and :class:`RecursivePipeline`
which offer additional functionality.
"""

from abc import ABC

from pyspark import keyword_only
from pyspark.ml.wrapper import JavaEstimator
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.pipeline import Pipeline, PipelineModel, Estimator, Transformer
from sparknlp.common import AnnotatorProperties
from sparknlp.internal import AnnotatorTransformer, RecursiveEstimator, RecursiveTransformer

from sparknlp.annotation import Annotation
import sparknlp.internal as _internal
