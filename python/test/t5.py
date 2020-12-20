import unittest

from test.util import SparkContextForTest
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import SentenceDetectorDLModel, T5Transformer


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

        sentence_detector = SentenceDetectorDLModel\
            .pretrained()\
            .setInputCols(["documents"])\
            .setOutputCol("questions")

        t5 = T5Transformer()\
            .load("/models/sparknlp/google_t5_small_ssm_nq")\
            .setInputCols(["questions"])\
            .setOutputCol("answers")\

        pipeline = Pipeline().setStages([document_assembler, sentence_detector, t5])
        results = pipeline.fit(data).transform(data)

        results.select("questions.result", "answers.result").show(truncate=False)


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

        t5 = T5Transformer() \
            .load("/models/sparknlp/t5_small") \
            .setTask("summarize:")\
            .setMaxOutputLength(200)\
            .setInputCols(["documents"]) \
            .setOutputCol("summaries")

        pipeline = Pipeline().setStages([document_assembler, t5])
        results = pipeline.fit(data).transform(data)

        results.select("summaries.result").show(truncate=False)
