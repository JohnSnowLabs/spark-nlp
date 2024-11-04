#  Copyright 2017-2024 John Snow Labs
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

import sparknlp
from test.util import SparkContextForTest


@pytest.mark.fast
class SparkNLPTestSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        html_df = sparknlp.read().html("https://www.wikipedia.org")
        html_df.show()
        assert html_df.select("html").count() > 0

        params = {"titleFontSize": "12"}
        html_params_df = sparknlp.read(params).html("https://www.wikipedia.org")
        html_params_df.show()

        assert html_params_df.select("html").count() > 0