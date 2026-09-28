#  Copyright 2017-2026 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os
import unittest

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkSessionForTest


class DolphinForDocumentParsingTestSetup(unittest.TestCase):
    """Runs against a Dolphin 1.5 export at ``DOLPHIN_ONNX_PATH`` (``scripts/dolphin/export.sh``);
    skips when it is absent. Expected text comes from ``reference_1.5.json``.
    """

    def setUp(self):
        self.model_path = os.environ.get("DOLPHIN_ONNX_PATH", "models/dolphin_onnx")
        if not os.path.exists(self.model_path):
            self.skipTest(
                f"Dolphin ONNX model not found at {self.model_path}; set DOLPHIN_ONNX_PATH")
        self.fixtures = os.path.join(os.getcwd(), "../src/test/resources/dolphin")
        self.spark = SparkSessionForTest.spark

    def reference(self, image, mode):
        with open(os.path.join(self.fixtures, "reference_1.5.json"), encoding="utf-8") as handle:
            cases = json.load(handle)["cases"]
        return next(c for c in cases if c["image"] == image and c["mode"] == mode)

    def image_df(self, file_name):
        return self.spark.read.format("image") \
            .option("dropInvalid", value=True) \
            .load("file://" + os.path.join(self.fixtures, file_name))

    def build_pipeline(self, **kwargs):
        image_assembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")
        dolphin = DolphinForDocumentParsing \
            .loadSavedModel(self.model_path, self.spark) \
            .setInputCols(["image_assembler"]) \
            .setOutputCol("elements")
        for key, value in kwargs.items():
            getattr(dolphin, "set" + key[0].upper() + key[1:])(value)
        return Pipeline(stages=[image_assembler, dolphin])

    def run_on(self, file_name, **kwargs):
        df = self.image_df(file_name)
        result = self.build_pipeline(**kwargs).fit(df).transform(df)
        return result.select("elements").collect()[0][0]


@pytest.mark.slow
class DolphinElementLevelTest(DolphinForDocumentParsingTestSetup):

    def test_parses_a_cropped_table_to_html(self):
        elements = self.run_on("table_1.jpeg", parsingMode="table")
        self.assertEqual(1, len(elements))
        self.assertTrue(elements[0].result.startswith("<table>"))
        self.assertEqual("Table", elements[0].metadata["elementType"])

    def test_table_carries_reader2table_compatible_json(self):
        elements = self.run_on("table_1.jpeg", parsingMode="table")
        table_json = elements[0].metadata.get("tableJson")
        self.assertIsNotNone(table_json)
        self.assertIn("rows", json.loads(table_json))

    def test_parses_a_cropped_text_line(self):
        elements = self.run_on("para_1.jpg", parsingMode="text")
        self.assertEqual(self.reference("para_1.jpg", "text")["output"], elements[0].result)

    def test_parses_a_formula_with_the_formula_prompt(self):
        elements = self.run_on("line_formula.jpeg", parsingMode="formula")
        self.assertEqual(self.reference("line_formula.jpeg", "formula")["output"],
                         elements[0].result)
        self.assertEqual("equ", elements[0].metadata["dolphinLabel"])


@pytest.mark.slow
class DolphinPageLevelTest(DolphinForDocumentParsingTestSetup):

    def test_detects_layout_without_transcribing(self):
        elements = self.run_on("page_1.jpeg", parsingMode="layout")
        self.assertGreater(len(elements), 1)
        self.assertTrue(all(e.result == "" for e in elements))
        self.assertTrue(all("bbox" in e.metadata for e in elements))

    def test_parses_a_page_into_ordered_elements(self):
        elements = self.run_on("page_1.jpeg", parsingMode="page")
        expected = self.reference("page_1.jpeg", "page")["elements"]
        self.assertEqual(
            [(e["reading_order"], e["label"], ",".join(map(str, e["bbox"])), e["text"])
             for e in expected],
            [(int(e.metadata["readingOrder"]), e.metadata["dolphinLabel"], e.metadata["bbox"],
              e.result) for e in elements])

    def test_assembles_markdown(self):
        elements = self.run_on("page_1.jpeg", parsingMode="page", outputFormat="markdown")
        self.assertEqual(1, len(elements))
        self.assertEqual(
            len(self.reference("page_1.jpeg", "page")["elements"]),
            int(elements[0].metadata["elementCount"]))


@pytest.mark.fast
class DolphinParamTest(unittest.TestCase):
    """Params and validation need no model, so these run in CI."""

    def test_defaults_match_the_scala_side(self):
        dolphin = DolphinForDocumentParsing()
        self.assertEqual("page", dolphin.getParsingMode())
        self.assertEqual("elements", dolphin.getOutputFormat())
        self.assertEqual(4, dolphin.getElementBatchSize())
        self.assertEqual(4096, dolphin.getMaxOutputLength())
        self.assertEqual(4096, dolphin.getLayoutMaxOutputLength())
        self.assertAlmostEqual(1.0, dolphin.getRepetitionPenalty())
        self.assertTrue(dolphin.getAdjustBoxEdges())
        self.assertEqual(3, dolphin.getMinElementSize())
        self.assertEqual(896, dolphin.getOrDefault("size"))
        self.assertEqual(2, dolphin.getOrDefault("resample"))

    def test_accepts_every_documented_parsing_mode(self):
        dolphin = DolphinForDocumentParsing()
        for mode in ["page", "layout", "table", "text", "formula", "code"]:
            self.assertEqual(mode, dolphin.setParsingMode(mode).getParsingMode())

    def test_rejects_an_unknown_parsing_mode(self):
        with self.assertRaises(ValueError) as context:
            DolphinForDocumentParsing().setParsingMode("pages")
        self.assertIn("parsingMode must be one of", str(context.exception))

    def test_rejects_an_unknown_output_format(self):
        with self.assertRaises(ValueError) as context:
            DolphinForDocumentParsing().setOutputFormat("html")
        self.assertIn("outputFormat must be one of", str(context.exception))

    def test_rejects_non_positive_sizes(self):
        with self.assertRaises(ValueError):
            DolphinForDocumentParsing().setElementBatchSize(0)
        with self.assertRaises(ValueError):
            DolphinForDocumentParsing().setMaxOutputLength(-1)


@pytest.mark.slow
class DolphinSerializationTest(DolphinForDocumentParsingTestSetup):

    def test_survives_a_save_load_round_trip(self):
        import shutil

        saved = "./tmp_dolphin_py_model"
        try:
            image_assembler = ImageAssembler() \
                .setInputCol("image") \
                .setOutputCol("image_assembler")
            dolphin = DolphinForDocumentParsing \
                .loadSavedModel(self.model_path, self.spark) \
                .setInputCols(["image_assembler"]) \
                .setOutputCol("elements") \
                .setParsingMode("text")

            df = self.image_df("para_1.jpg")
            before = Pipeline(stages=[image_assembler, dolphin]) \
                .fit(df).transform(df).select("elements").collect()[0][0]

            dolphin.write().overwrite().save(saved)
            reloaded = DolphinForDocumentParsing.load(saved) \
                .setInputCols(["image_assembler"]) \
                .setOutputCol("elements")
            after = Pipeline(stages=[image_assembler, reloaded]) \
                .fit(df).transform(df).select("elements").collect()[0][0]

            self.assertEqual([e.result for e in before], [e.result for e in after])
            self.assertEqual("text", reloaded.getParsingMode())
        finally:
            shutil.rmtree(saved, ignore_errors=True)
