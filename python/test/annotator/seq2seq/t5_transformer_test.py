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

from sparknlp.annotator import *
from sparknlp.base import *
from test.util import SparkContextForTest


@pytest.mark.slow
class T5TransformerQATestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, "Which is the capital of France? Who was the first president of USA?"],
            [1, "Which is the capital of Bulgaria ?"],
            [2, "Who is Donald Trump?"]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        sentence_detector = SentenceDetectorDLModel \
            .pretrained() \
            .setInputCols(["documents"]) \
            .setOutputCol("questions")

        t5 = T5Transformer.pretrained() \
            .setInputCols(["questions"]) \
            .setOutputCol("answers")

        pipeline = Pipeline().setStages([document_assembler, sentence_detector, t5])
        results = pipeline.fit(data).transform(data)

        results.select("questions.result", "answers.result").show(truncate=False)


@pytest.mark.slow
class T5TransformerSummaryTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """
            Heat oven to 200C/180C fan/gas 6. Line each hole of a 12-hole muffin tin with a thin strip of baking 
            parchment across the middle that’s long enough so the ends stick out a centimetre or two – use a dab of
             butter to stick in place. Roll out two thirds of the pastry on a lightly floured surface and stamp out 
             12 x 10cm circles (you may need to re-roll trimmings). Press a circle into each hole to line.
             
            Sprinkle 1 tsp of breadcrumbs into the base of each pie. Tip the rest of the crumbs into a mixing bowl. 
            Squeeze in the sausage meat, discarding the skins, along with the bacon, mace, pepper, sage and just a 
            little salt. Get your hands in and mash and squish everything together until the breadcrumbs have just 
            about disappeared. Divide mixture between the holes, packing in firmly and shaping to a dome in the middle.
             
            Roll out the remaining pastry and stamp out 12 x 7cm circles. Brush with a little egg and add a top to 
            each pie, egg-side down to stick, carefully pressing pastry edges together to seal. Brush with more egg 
            (don’t throw away leftovers) and sprinkle with sesame seeds. Bake for 30 mins until golden then carefully 
            remove the pies from the tin, using the parchment ends to help you lift them out. Sit on a parchment lined 
            baking tray, brush all round the sides with more egg and put back in the oven for 8 mins. Cool completely 
            then eat with piccalilli, or your favourite pickle.             
            """.strip().replace("\n", " ")]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        t5 = T5Transformer.pretrained() \
            .setTask("summarize:") \
            .setMaxOutputLength(200) \
            .setInputCols(["documents"]) \
            .setOutputCol("summaries")

        pipeline = Pipeline().setStages([document_assembler, t5])
        results = pipeline.fit(data).transform(data)

        results.select("summaries.result").show(truncate=False)


@pytest.mark.slow
class T5TransformerSummaryWithRepetitionPenaltyTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness
              of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons
             of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until
             browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside.
             Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until
             softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the
             time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a
              large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush
             with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C
             /fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6,
              uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.
              """.strip().replace("\n", " ")]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        t5 = T5Transformer.pretrained() \
            .setTask("summarize:") \
            .setMaxOutputLength(50) \
            .setInputCols(["documents"]) \
            .setOutputCol("summaries") \
            .setRepetitionPenalty(2)

        pipeline = Pipeline().setStages([document_assembler, t5])
        results = pipeline.fit(data).transform(data)

        results.select("summaries.result").show(truncate=False)


@pytest.mark.slow
class T5TransformerSummaryWithSamplingAndDeactivatedTopKTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness
              of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons
             of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until
             browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside.
             Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until
             softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the
             time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a
              large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush
             with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C
             /fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6,
              uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.
              """.strip().replace("\n", " ")]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        t5 = T5Transformer.pretrained() \
            .setTask("summarize:") \
            .setMaxOutputLength(50) \
            .setDoSample(True) \
            .setInputCols(["documents"]) \
            .setOutputCol("summaries") \
            .setTopK(0)

        pipeline = Pipeline().setStages([document_assembler, t5])
        results = pipeline.fit(data).transform(data)

        results.select("summaries.result").show(truncate=False)


@pytest.mark.slow
class T5TransformerSummaryWithSamplingAndTemperatureTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness
              of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons
             of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until
             browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside.
             Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until
             softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the
             time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a
              large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush
             with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C
             /fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6,
              uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.
              """.strip().replace("\n", " ")]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        t5 = T5Transformer.pretrained() \
            .setTask("summarize:") \
            .setMaxOutputLength(50) \
            .setDoSample(True) \
            .setInputCols(["documents"]) \
            .setOutputCol("summaries") \
            .setTopK(50) \
            .setTemperature(0.7)

        pipeline = Pipeline().setStages([document_assembler, t5])
        results = pipeline.fit(data).transform(data)

        results.select("summaries.result").show(truncate=False)


@pytest.mark.slow
class T5TransformerSummaryWithSamplingAndTopPTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness
              of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons
             of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until
             browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside.
             Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until
             softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the
             time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a
              large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush
             with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C
             /fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6,
              uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.
              """.strip().replace("\n", " ")]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        t5 = T5Transformer.pretrained() \
            .setTask("summarize:") \
            .setMaxOutputLength(50) \
            .setDoSample(True) \
            .setInputCols(["documents"]) \
            .setOutputCol("summaries") \
            .setTopK(0) \
            .setTopP(0.7)

        pipeline = Pipeline().setStages([document_assembler, t5])
        results = pipeline.fit(data).transform(data)

        results.select("summaries.result").show(truncate=False)


@pytest.mark.slow
class T5TransformerSummaryWithSamplingTestSpec(unittest.TestCase):
    def setUp(self):
        self.spark = SparkContextForTest.spark

    def runTest(self):
        data = self.spark.createDataFrame([
            [1, """Preheat the oven to 220°C/ fan200°C/gas 7. Trim the lamb fillet of fat and cut into slices the thickness
              of a chop. Cut the kidneys in half and snip out the white core. Melt a knob of dripping or 2 tablespoons
             of vegetable oil in a heavy large pan. Fry the lamb fillet in batches for 3-4 minutes, turning once, until
             browned. Set aside. Fry the kidneys and cook for 1-2 minutes, turning once, until browned. Set aside.
             Wipe the pan with kitchen paper, then add the butter. Add the onions and fry for about 10 minutes until
             softened. Sprinkle in the flour and stir well for 1 minute. Gradually pour in the stock, stirring all the
             time to avoid lumps. Add the herbs. Stir the lamb and kidneys into the onions. Season well. Transfer to a
              large 2.5-litre casserole. Slice the peeled potatoes thinly and arrange on top in overlapping rows. Brush
             with melted butter and season. Cover and bake for 30 minutes. Reduce the oven temperature to 160°C
             /fan140°C/gas 3 and cook for a further 2 hours. Then increase the oven temperature to 200°C/ fan180°C/gas 6,
              uncover, and brush the potatoes with more butter. Cook uncovered for 15-20 minutes, or until golden.
              """.strip().replace("\n", " ")]]).toDF("id", "text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("documents")

        t5 = T5Transformer.pretrained() \
            .setTask("summarize:") \
            .setMaxOutputLength(50) \
            .setDoSample(True) \
            .setInputCols(["documents"]) \
            .setOutputCol("summaries")

        pipeline = Pipeline().setStages([document_assembler, t5])
        results = pipeline.fit(data).transform(data)

        results.select("summaries.result").show(truncate=False)

