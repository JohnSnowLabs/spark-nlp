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

from sparknlp.base import *
from sparknlp.annotator import *
from test.util import SparkSessionForTest


@pytest.mark.fast
class TableAssemblerBasicTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.json_table_data = '{"header": ["Name", "Age"], "rows": [["John", "30"], ["Jane", "25"]]}'
        self.json_data_set = self.spark.createDataFrame([[self.json_table_data]]) \
            .toDF("table_source")

    def runTest(self):
        document_assembler = DocumentAssembler() \
            .setInputCol("table_source") \
            .setOutputCol("document")

        table_assembler = TableAssembler() \
            .setInputFormat("json") \
            .setInputCols(["document"]) \
            .setOutputCol("table")

        finisher = Finisher() \
            .setInputCols(["table"]) \
            .setOutputAsArray(True) \
            .setCleanAnnotations(False) \
            .setOutputCols(["output"])

        pipeline = Pipeline().setStages([document_assembler, table_assembler, finisher])
        result_df = pipeline.fit(self.json_data_set).transform(self.json_data_set)

        output = result_df.select("output").collect()[0]

        self.assertGreater(len(output), 0)


@pytest.mark.slow
class TableAssemblerEndToEndTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.json_table_data1 = '{"header": ["Name", "Salary", "Country"], "rows": [["Elon Musk", "100000000", "USA"], ["Jeff Bezos", "95000000", "USA"], ["Bill Gates", "90000000", "USA"]]}'
        self.json_table_data2 = '{"header": ["Product", "Price", "Stock"], "rows": [["Laptop", "1200", "50"], ["Phone", "800", "100"], ["Tablet", "600", "75"]]}'
        self.csv_table_data = """Name, Age, Department
Alice Johnson, 28, Engineering
Bob Smith, 35, Marketing
Carol Williams, 42, Finance
David Brown, 31, Engineering"""

    def runTest(self):
        json_data_set1 = self.spark.createDataFrame([[self.json_table_data1]]).toDF("table_source")
        json_data_set2 = self.spark.createDataFrame([[self.json_table_data2]]).toDF("table_source")

        document_assembler = DocumentAssembler() \
            .setInputCol("table_source") \
            .setOutputCol("document")

        table_assembler_json = TableAssembler() \
            .setInputFormat("json") \
            .setInputCols(["document"]) \
            .setOutputCol("table")

        finisher = Finisher() \
            .setInputCols(["table"]) \
            .setOutputAsArray(True) \
            .setCleanAnnotations(False) \
            .setOutputCols(["output"])

        pipeline_json = Pipeline().setStages([document_assembler, table_assembler_json, finisher])

        result_df1 = pipeline_json.fit(json_data_set1).transform(json_data_set1)
        result_df2 = pipeline_json.fit(json_data_set2).transform(json_data_set2)

        output1 = result_df1.select("output").collect()[0]
        output2 = result_df2.select("output").collect()[0]

        self.assertGreater(len(output1), 0, "First JSON table should have output")
        self.assertGreater(len(output2), 0, "Second JSON table should have output")

        csv_data_set = self.spark.createDataFrame([[self.csv_table_data]]).toDF("table_source")

        table_assembler_csv = TableAssembler() \
            .setInputFormat("csv") \
            .setInputCols(["document"]) \
            .setOutputCol("table")

        pipeline_csv = Pipeline().setStages([document_assembler, table_assembler_csv, finisher])
        result_df_csv = pipeline_csv.fit(csv_data_set).transform(csv_data_set)

        output_csv = result_df_csv.select("output").collect()[0]

        self.assertGreater(len(output_csv), 0, "CSV table should have output")
        self.assertTrue(len(output1) > 0 and len(output2) > 0 and len(output_csv) > 0,
                        "All three tables should be processed successfully")