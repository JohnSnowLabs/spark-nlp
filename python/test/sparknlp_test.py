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
import os
import sparknlp
from test.util import SparkContextForTest


@pytest.mark.fast
class SparkNLPTestHTMLRealTimeSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        html_df = sparknlp.read().html("https://www.wikipedia.org")
        html_df.show()
        assert html_df.select("html").count() > 0

        params = {"titleFontSize": "12"}
        html_params_df = sparknlp.read(params).html("https://www.wikipedia.org")
        html_params_df.show()

        self.assertTrue(html_params_df.select("html").count() > 0)


@pytest.mark.fast
class SparkNLPTestHTMLFilesSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data
        self.html_file = f"file:///{os.getcwd()}/../src/test/resources/reader/html/fake-html.html"

    def runTest(self):
        html_df = sparknlp.read().html(self.html_file)
        html_df.show()

        self.assertTrue(html_df.select("html").count() > 0)


@pytest.mark.fast
class SparkNLPTestHTMLValidationSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data

    def runTest(self):
        with pytest.raises(TypeError, match="htmlPath must be a string or a list of strings"):
            sparknlp.read().html(123)


@pytest.mark.fast
class SparkNLPTestEmailFilesSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data
        self.email_file = f"file:///{os.getcwd()}/../src/test/resources/reader/email/test-several-attachments.eml"

    def runTest(self):
        email_df = sparknlp.read().email(self.email_file)
        email_df.show()

        self.assertTrue(email_df.select("email").count() > 0)

@pytest.mark.fast
class SparkNLPTestWordFilesSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data
        self.word_file = f"file:///{os.getcwd()}/../src/test/resources/reader/doc/contains-pictures.docx"

    def runTest(self):
        word_df = sparknlp.read().doc(self.word_file)
        word_df.show()

        self.assertTrue(word_df.select("doc").count() > 0)

@pytest.mark.fast
class SparkNLPTestExcelFilesSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data
        self.excel_file = f"file:///{os.getcwd()}/../src/test/resources/reader/xls/vodafone.xlsx"

    def runTest(self):
        excel_df = sparknlp.read().xls(self.excel_file)
        excel_df.show()

        self.assertTrue(excel_df.select("xls").count() > 0)

@pytest.mark.fast
class SparkNLPTestPowerPointFilesSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data
        self.ppt_file = f"file:///{os.getcwd()}/../src/test/resources/reader/ppt"

    def runTest(self):
        ppt_df = sparknlp.read().ppt(self.ppt_file)
        ppt_df.show()

        self.assertTrue(ppt_df.select("ppt").count() > 0)

@pytest.mark.fast
class SparkNLPTestTXTFilesSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data
        self.txt_file = f"file:///{os.getcwd()}/../src/test/resources/reader/txt/simple-text.txt"

    def runTest(self):
        txt_df = sparknlp.read().txt(self.txt_file)
        txt_df.show()

        self.assertTrue(txt_df.select("txt").count() > 0)


@pytest.mark.fast
class SparkNLPTestXMLFilesSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data
        self.xml_files = f"file:///{os.getcwd()}/../src/test/resources/reader/xml"

    def runTest(self):
        xml_df = sparknlp.read().xml(self.xml_files)
        xml_df.show()

        self.assertTrue(xml_df.select("xml").count() > 0)

@pytest.mark.fast
class SparkNLPTestMdFilesSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data
        self.md_file = f"file:///{os.getcwd()}/../src/test/resources/reader/md/simple.md"

    def runTest(self):
        md_df = sparknlp.read().md(self.md_file)
        md_df.show()

        self.assertTrue(md_df.select("md").count() > 0)


@pytest.mark.fast
class SparkNLPTestMdContentSpec(unittest.TestCase):

    def setUp(self):
        self.data = SparkContextForTest.data
        self.content = """\
                        # Shopping List
                         - Milk
                         - Bread
                         - Eggs
                        """

    def runTest(self):
        md_df = sparknlp.read().md(text=self.content)
        md_df.show()

        self.assertTrue(md_df.select("md").count() > 0)