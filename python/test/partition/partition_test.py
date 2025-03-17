#  Copyright 2017-2025 John Snow Labs
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
import os
import unittest
import pytest
from sparknlp.partition.partition import Partition


@pytest.mark.fast
class PartitionTextTesSpec(unittest.TestCase):

    def setUp(self):
        self.txt_directory = f"file:///{os.getcwd()}/../src/test/resources/reader/txt"

    def runTest(self):
        text_df = Partition(content_type = "text/plain").partition(self.txt_directory)
        text_file_df = Partition().partition(f"{self.txt_directory}/simple-text.txt")

        self.assertTrue(text_df.select("txt").count() > 0)
        self.assertTrue(text_file_df.select("txt").count() > 0)


@pytest.mark.fast
class PartitionWordTesSpec(unittest.TestCase):

    def setUp(self):
        self.word_directory = f"file:///{os.getcwd()}/../src/test/resources/reader/doc"

    def runTest(self):
        doc_df = Partition(content_type = "application/msword").partition(self.word_directory)
        doc_file_df = Partition().partition(f"{self.word_directory}/fake_table.docx")

        self.assertTrue(doc_df.select("doc").count() > 0)
        self.assertTrue(doc_file_df.select("doc").count() > 0)


@pytest.mark.fast
class PartitionExcelTesSpec(unittest.TestCase):

    def setUp(self):
        self.excel_directory = f"file:///{os.getcwd()}/../src/test/resources/reader/xls"

    def runTest(self):
        xls_df = Partition(content_type = "application/vnd.ms-excel").partition(self.excel_directory)
        xls_file_df = Partition().partition(f"{self.excel_directory}/vodafone.xlsx")

        self.assertTrue(xls_df.select("xls").count() > 0)
        self.assertTrue(xls_file_df.select("xls").count() > 0)


@pytest.mark.fast
class PartitionPowerPointTesSpec(unittest.TestCase):

    def setUp(self):
        self.ppt_directory = f"file:///{os.getcwd()}/../src/test/resources/reader/ppt"

    def runTest(self):
        ppt_df = Partition(content_type = "application/vnd.ms-powerpoint").partition(self.ppt_directory)
        ppt_file_df = Partition().partition(f"{self.ppt_directory}/fake-power-point.pptx")

        self.assertTrue(ppt_df.select("ppt").count() > 0)
        self.assertTrue(ppt_file_df.select("ppt").count() > 0)


@pytest.mark.fast
class PartitionEmailTesSpec(unittest.TestCase):

    def setUp(self):
        self.eml_directory = f"file:///{os.getcwd()}/../src/test/resources/reader/email"

    def runTest(self):
        eml_df = Partition(content_type = "message/rfc822").partition(self.eml_directory)
        eml_file_df = Partition().partition(f"{self.eml_directory}/test-several-attachments.eml")

        self.assertTrue(eml_df.select("email").count() > 0)
        self.assertTrue(eml_file_df.select("email").count() > 0)


@pytest.mark.fast
class PartitionHtmlTesSpec(unittest.TestCase):

    def setUp(self):
        self.html_directory = f"file:///{os.getcwd()}/../src/test/resources/reader/html"

    def runTest(self):
        html_df = Partition(content_type = "text/html").partition(self.html_directory)
        html_file_df = Partition().partition(f"{self.html_directory}/fake-html.html")

        self.assertTrue(html_df.select("html").count() > 0)
        self.assertTrue(html_file_df.select("html").count() > 0)


@pytest.mark.fast
class PartitionUrlTesSpec(unittest.TestCase):

    def runTest(self):
        url_df = Partition().partition("https://www.wikipedia.org")
        urls_df = Partition().partition(["https://www.wikipedia.org", "https://example.com/"])

        self.assertTrue(url_df.select("html").count() > 0)
        self.assertTrue(urls_df.select("html").count() > 0)


@pytest.mark.fast
class PartitionPdfTesSpec(unittest.TestCase):

    def setUp(self):
        self.html_directory = f"file:///{os.getcwd()}/../src/test/resources/reader/pdf"

    def runTest(self):
        pdf_df = Partition(content_type = "application/pdf").partition(self.html_directory)
        pdf_file_df = Partition().partition(f"{self.html_directory}/text_3_pages.pdf")

        self.assertTrue(pdf_df.select("text").count() > 0)
        self.assertTrue(pdf_file_df.select("text").count() > 0)
