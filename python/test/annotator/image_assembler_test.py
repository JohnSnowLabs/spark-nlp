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
import unittest

import pytest
import os

from sparknlp.base import *
from test.util import SparkSessionForTest


@pytest.mark.fast
class ImageAssemblerTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkSessionForTest.spark.read.format("image")\
            .load(path="file:///" + os.getcwd() + "/../src/test/resources/image/")

    def runTest(self):
        image_assembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")
        assembled = image_assembler.transform(self.data)
        assembled.select("image_assembler.result").show()
        assembled.select("image_assembler.origin").show()
        assembled.select("image_assembler.metadata").show()
