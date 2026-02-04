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
import os
import unittest
from collections.abc import Sequence

import pytest

from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline
from test.util import SparkSessionForTest, SparkContextForTest


class LightPipelineTextSetUp(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.text = "This is a text input"
        self.textDataSet = self.spark.createDataFrame([[self.text]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        regex_tok = RegexTokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        pipeline = Pipeline().setStages([document_assembler, regex_tok])
        self.model = pipeline.fit(self.textDataSet)


@pytest.mark.fast
class LightPipelineTextInputTest(LightPipelineTextSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        annotations_result = light_pipeline.fullAnnotate(self.text)

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result["document"]) > 0)
            self.assertTrue(len(result["token"]) > 0)

        texts = [self.text, self.text]
        annotations_result = light_pipeline.fullAnnotate(texts)

        self.assertEqual(len(annotations_result), len(texts))
        for result in annotations_result:
            self.assertTrue(len(result["document"]) > 0)
            self.assertTrue(len(result["token"]) > 0)


@pytest.mark.fast
class LightPipelineNoisyTextInputTest(LightPipelineTextSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.noisy_text = "I am still so sick  http://myloc.me/22zV"

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        annotations_result = light_pipeline.fullAnnotate(self.text)

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result["document"]) > 0)
            self.assertTrue(len(result["token"]) > 0)

        annotations_result = light_pipeline.fullAnnotate(self.noisy_text)

        self.assertEqual(len(annotations_result), 1)
        for result in annotations_result:
            self.assertTrue(len(result["document"]) > 0)
            self.assertTrue(len(result["token"]) > 0)

        texts = [self.noisy_text, self.text]
        annotations_result = light_pipeline.fullAnnotate(texts)

        self.assertEqual(len(annotations_result), len(texts))
        for result in annotations_result:
            self.assertTrue(len(result["document"]) > 0)
            self.assertTrue(len(result["token"]) > 0)


class LightPipelineImageSetUp(unittest.TestCase):

    def setUp(self):
        self.images_path = os.getcwd() + "/../src/test/resources/image/"
        self.data = SparkSessionForTest.spark.read.format("image") \
            .load(path=self.images_path)

        image_assembler = ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")

        image_classifier = ViTForImageClassification \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("class")

        pipeline = Pipeline(stages=[
            image_assembler,
            image_classifier,
        ])

        self.vit_model = pipeline.fit(self.data)


@pytest.mark.slow
class LightPipelineImageTest(LightPipelineImageSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        image = self.images_path + "hippopotamus.JPEG"
        light_pipeline = LightPipeline(self.vit_model)

        annotations_result = light_pipeline.fullAnnotateImage(image)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

        annotations_result = light_pipeline.fullAnnotate(image)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["class"]) > 0)


@pytest.mark.slow
class LightPipelineImagesInputTest(LightPipelineImageSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        images = [self.images_path + "hippopotamus.JPEG", self.images_path + "egyptian_cat.jpeg"]
        light_pipeline = LightPipeline(self.vit_model)

        annotations_result = light_pipeline.fullAnnotate(images)

        self.assertEqual(len(annotations_result), len(images))
        self.assertAnnotations(annotations_result)

        annotations_result = light_pipeline.fullAnnotate(images)

        self.assertEqual(len(annotations_result), len(images))
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["image_assembler"]) > 0)
            self.assertTrue(len(result["class"]) > 0)


class LightPipelineAudioBase(unittest.TestCase):
    def setUp(self):
        audio_json = os.getcwd() + "/../src/test/resources/audio/json/audio_floats.json"
        audio_csv = os.getcwd() + "/../src/test/resources/audio/csv/audio_floats.csv"
        self.data = SparkSessionForTest.spark.read.option("inferSchema", value=True).json(audio_json) \
            .select(col("float_array").cast("array<float>").alias("audio_content"))
        self.audio_data = list()
        audio_file = open(audio_csv, 'r')
        csv_lines = audio_file.readlines()
        for csv_line in csv_lines:
            clean_line = float(csv_line.split(',')[0])
            self.audio_data.append(clean_line)

        audio_assembler = AudioAssembler() \
            .setInputCol("audio_content") \
            .setOutputCol("audio_assembler")

        speech_to_text = Wav2Vec2ForCTC \
            .pretrained() \
            .setInputCols("audio_assembler") \
            .setOutputCol("text")

        pipeline = Pipeline(stages=[audio_assembler, speech_to_text])
        self.model = pipeline.fit(self.data)


@pytest.mark.slow
class LightPipelineAudioInputTest(LightPipelineAudioBase):

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        annotations_result = light_pipeline.fullAnnotate(self.audio_data)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

        self.audios = [self.audio_data, self.audio_data]
        annotations_result = light_pipeline.fullAnnotate(self.audios)

        self.assertEqual(len(annotations_result), 2)
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["audio_assembler"]) > 0)
            self.assertTrue(len(result["text"]) > 0)


@pytest.mark.slow
class LightPipelineTapasInputTest(unittest.TestCase):

    def setUp(self):
        table_json_source = os.getcwd() + "/../src/test/resources/tapas/rich_people.json"
        with open(table_json_source, "rt") as F:
            self.table = "".join(F.readlines())

        self.question1 = "Who earns 100,000,000?"
        self.question2 = "How much people earn?"
        self.data = SparkContextForTest.spark.createDataFrame([
            [self.table, self.question1],
            [self.table, self.question2]
        ]).toDF("table_json", "questions")

    def runTest(self):
        document_assembler = MultiDocumentAssembler() \
            .setInputCols("table_json", "questions") \
            .setOutputCols("document_questions", "document_table")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document_questions"]) \
            .setOutputCol("questions")

        table_assembler = TableAssembler() \
            .setInputCols(["document_table"]) \
            .setOutputCol("table")

        tapas = TapasForQuestionAnswering() \
            .pretrained() \
            .setMaxSentenceLength(512) \
            .setInputCols(["questions", "table"]) \
            .setOutputCol("answers")

        pipeline = Pipeline(stages=[
            document_assembler,
            sentence_detector,
            table_assembler,
            tapas
        ])

        model = pipeline.fit(self.data)

        light_pipeline = LightPipeline(model)
        annotations_result = light_pipeline.fullAnnotate(self.question1, self.table)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

        questions = [self.question1, self.question2]
        annotations_result = light_pipeline.fullAnnotate( target = questions, optional_target = [self.table, self.table])

        self.assertEqual(len(annotations_result), len(questions))
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["document_questions"]) > 0)
            self.assertTrue(len(result["document_table"]) > 0)
            self.assertTrue(len(result["questions"]) > 0)
            self.assertTrue(len(result["table"]) > 0)
            self.assertTrue(len(result["answers"]) > 0)


@pytest.mark.slow
class LightPipelineQAInputTest(unittest.TestCase):

    def setUp(self):
        self.question = "What's my name?"
        self.context = "My name is Clara and I live in Berkeley."
        self.data = SparkContextForTest.spark.createDataFrame([[self.question, self.context]]) \
            .toDF("question", "context")

    def runTest(self):
        document_assembler = MultiDocumentAssembler() \
            .setInputCols(["question", "context"]) \
            .setOutputCols(["document_question", "document_context"])

        qa_classifier = DistilBertForQuestionAnswering.pretrained() \
            .setInputCols(["document_question", 'document_context']) \
            .setOutputCol("answer")

        pipeline = Pipeline().setStages([
            document_assembler,
            qa_classifier
        ])

        model = pipeline.fit(self.data)

        light_pipeline = LightPipeline(model)
        annotations_result = light_pipeline.fullAnnotate(self.question, self.context)

        self.assertEqual(len(annotations_result), 1)
        self.assertAnnotations(annotations_result)

        questions = [self.question, self.question]
        contexts = [self.context, self.context]
        annotations_result = light_pipeline.fullAnnotate(questions, contexts)

        self.assertEqual(len(annotations_result), 2)
        self.assertAnnotations(annotations_result)

    def assertAnnotations(self, annotations_result):
        for result in annotations_result:
            self.assertTrue(len(result["document_question"]) > 0)
            self.assertTrue(len(result["document_context"]) > 0)
            self.assertTrue(len(result["answer"]) > 0)


@pytest.mark.fast
class LightPipelineWrongInputTest(LightPipelineTextSetUp, unittest.TestCase):

    def setUp(self):
        super().setUp()

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        with self.assertRaises(TypeError):
            light_pipeline.fullAnnotate(1)

        with self.assertRaises(TypeError):
            light_pipeline.fullAnnotate([1, 2])

        with self.assertRaises(TypeError):
            light_pipeline.fullAnnotate({"key": "value"})


@pytest.mark.fast
class LightPipelineWrongInputColsTest(unittest.TestCase):

    def setUp(self):
        self.sample_text = "I was born in 1990 ."
        self.data = SparkSessionForTest.spark.createDataFrame([[self.sample_text]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentence_detector = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("my_token")

        regex_matcher = RegexMatcher() \
            .setExternalRules(os.getcwd() + "/../src/test/resources/regex-matcher/rules.txt",  ",") \
            .setInputCols(["my_token"]) \
            .setOutputCol("regex") \
            .setStrategy("MATCH_ALL")

        self.pipeline = Pipeline().setStages([document_assembler, sentence_detector, tokenizer, regex_matcher])

    def runTest(self):
        model = self.pipeline.fit(self.data)

        light_model = LightPipeline(model)

        with self.assertRaises(TypeError):
            light_model.fullAnnotate(self.sample_text)

        with self.assertRaises(TypeError):
            light_model.annotate(self.sample_text)


@pytest.mark.slow
class LightPipelineWithEntitiesTest(unittest.TestCase):
    """Tests LightPipeline with embeddings parsing enabled, ID input, and entity recognition."""

    def setUp(self):
        self.pipeline = PretrainedPipeline("onto_recognize_entities_bert_tiny", lang="en")

        self.texts = [
            "Barack Obama was born in Hawaii and served as President of the United States.",
            "John Snow Labs is based in Delaware and builds AI for healthcare."
        ]
        self.ids = [1001, 1002]

    def runTest(self):
        light_pipeline = LightPipeline(self.pipeline.model, parse_embeddings=True)

        results = light_pipeline.fullAnnotate(self.ids, self.texts)
        self.assertEqual(len(results), len(self.texts))

        for i, res in enumerate(results):

            self.assertIn("doc_id", res)
            self.assertIn("token", res)
            self.assertIn("ner", res)
            self.assertIn("embeddings", res)

            colid = res["doc_id"][0]
            val = colid.result if hasattr(colid, "result") else colid
            self.assertEqual(val, str(self.ids[i]))

            ner_tags = [a.result for a in res["ner"] if hasattr(a, "result")]
            self.assertGreater(len(ner_tags), 0, "Expected at least one entity label")

            emb_anns = [a for a in res["embeddings"] if hasattr(a, "embeddings")]
            self.assertGreater(len(emb_anns), 0, "Expected embeddings annotations to be present")

        annotated = light_pipeline.annotate(self.ids, self.texts)
        self.assertEqual(len(annotated), len(self.texts))

        for i, res in enumerate(annotated):
            self.assertIn("doc_id", res)
            self.assertIn("ner", res)
            self.assertIn("embeddings", res)
            self.assertTrue(any(str(self.ids[i]) in s for s in res["doc_id"]))
            self.assertGreater(len(res["ner"]), 0, "Expected non-empty NER results")
            self.assertGreater(len(res["embeddings"]), 0, "Expected embeddings data")


class LightPipelineTextBase(unittest.TestCase):
    """Shared setup for text-based LightPipeline tests."""

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.text = "This is a text input"
        self.texts = ["This is text one", "This is text two"]
        self.ids = [1, 2]
        self.textDataSet = self.spark.createDataFrame([[self.text]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        regex_tok = RegexTokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        pipeline = Pipeline().setStages([document_assembler, regex_tok])
        self.model = pipeline.fit(self.textDataSet)
        self.light_pipeline = LightPipeline(self.model)


@pytest.mark.fast
class LightPipelineFullAnnotateMetadataTest(LightPipelineTextBase):
    """Tests LightPipeline metadata for fullAnnotate."""

    def runTest(self):
        light_pipeline = LightPipeline(self.model, parse_embeddings=False)
        result = light_pipeline.fullAnnotate(self.texts)
        expected_metadata = {"sentence": "0"}

        doc_metadata = result[0]["document"][0].metadata
        self.assertEqual(dict(doc_metadata), expected_metadata)

        result_id = light_pipeline.fullAnnotate(self.ids, self.texts)
        expected_metadata_id = {"sentence": "0", "id": "1"}
        doc_metadata = result_id[0]["document"][0].metadata
        self.assertEqual(dict(doc_metadata), expected_metadata_id)


@pytest.mark.fast
class LightPipelineWithPostOutputTest(LightPipelineTextBase):
    """Tests LightPipeline output types (list vs string) for annotate and fullAnnotate."""

    def runTest(self):
        light_pipeline = LightPipeline(self.model, parse_embeddings=False)

        # Helper for all sequence checks
        def assert_sequence(obj, msg):
            self.assertTrue(
                isinstance(obj, Sequence) and not isinstance(obj, str),
                f"{msg} (got {type(obj)})"
            )

        results = light_pipeline.annotate(self.texts)[0]['token']
        assert_sequence(results, "Expected sequence when annotating list of texts")

        results = light_pipeline.annotate(self.texts[1])['token']
        assert_sequence(results, "Expected sequence when annotating single text")

        token = light_pipeline.annotate(self.texts[1])['token'][0]
        self.assertIsInstance(token, str, "Expected string when accessing single token")

        results = light_pipeline.fullAnnotate(self.texts)[1]['token']
        assert_sequence(results, "Expected sequence of Annotations when using fullAnnotate(list)")

        token_anno = light_pipeline.fullAnnotate(self.texts)[1]['token'][0]
        self.assertTrue(
            hasattr(token_anno, "result") or "Annotation" in str(type(token_anno)),
            f"Expected Annotation-like object when accessing single token, got {type(token_anno)}"
        )


@pytest.mark.fast
class LightPipelineAnnotateTextInputTest(LightPipelineTextBase):
    """Covers all combinations for annotate(text | list[str]) input."""

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        valid_cases = [
            ("annotate(text)", (self.text,), dict),
            ("annotate(list[str])", (self.texts,), list),
            ("annotate(target=str)", (), dict, {"target": self.text}),
            ("annotate(target=list[str])", (), list, {"target": self.texts}),
        ]

        for label, args, expected_type, *kw in valid_cases:
            kwargs = kw[0] if kw else {}
            with self.subTest(label=label):
                result = light_pipeline.annotate(*args, **kwargs)
                self.assertIsInstance(result, expected_type,
                                      f"{label} should return {expected_type.__name__}")

        invalid_cases = [
            ("annotate(str, list[str])", (self.text, self.texts)),
            ("annotate(non-str)", (1,)),
            ("annotate(list[int])", ([1, 2],)),
        ]

        for label, args in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaises(TypeError, msg=f"{label} should raise TypeError"):
                    light_pipeline.annotate(*args)


@pytest.mark.fast
class LightPipelineFullAnnotateTextInputTest(LightPipelineTextBase):
    """Covers all combinations for fullAnnotate(text | list[str]) input."""

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        valid_cases = [
            ("fullAnnotate(str)", (self.text,), list),
            ("fullAnnotate(list[str])", (self.texts,), list),
            ("fullAnnotate(target=str)", (), list, {"target": self.text}),
            ("fullAnnotate(target=list[str])", (), list, {"target": self.texts}),
        ]

        for label, args, expected_type, *kw in valid_cases:
            kwargs = kw[0] if kw else {}
            with self.subTest(label=label):
                result = light_pipeline.fullAnnotate(*args, **kwargs)
                self.assertIsInstance(result, expected_type,
                                      f"{label} should return {expected_type.__name__}")
                # Ensure each item is a dict containing Annotation lists
                for item in result:
                    self.assertIsInstance(item, dict)
                    for v in item.values():
                        if v:
                            self.assertTrue(
                                all(isinstance(a, Annotation) for a in v),
                                f"{label} expected list of Annotation objects"
                            )

        invalid_cases = [
            ("fullAnnotate(str, list[str])", (self.text, self.texts)),
            ("fullAnnotate(non-str)", (1,)),
            ("fullAnnotate(list[int])", ([1, 2],)),
        ]

        for label, args in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaises(TypeError, msg=f"{label} should raise TypeError"):
                    light_pipeline.fullAnnotate(*args)


class LightPipelineIdsTextsBase(unittest.TestCase):
    """Shared setup for (ids, texts) LightPipeline tests."""

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.text = "This is a text input"
        self.texts = [
            "Barack Obama was born in Hawaii.",
            "John Snow Labs builds AI for healthcare."
        ]
        self.ids = [101, 102]

        self.textDataSet = self.spark.createDataFrame([[self.text]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        regex_tok = RegexTokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        pipeline = Pipeline().setStages([document_assembler, regex_tok])
        self.model = pipeline.fit(self.textDataSet)


@pytest.mark.fast
class LightPipelineAnnotateIdsTextsInputTest(LightPipelineIdsTextsBase):
    """Tests annotate(ids, texts) and keyword equivalents."""

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        valid_cases = [
            ("annotate(ids, texts)", (self.ids, self.texts), {}, list),
            ("annotate(target=ids, optional_target=texts)",
             (), {"target": self.ids, "optional_target": self.texts}, list),
        ]

        for label, args, kwargs, expected_type in valid_cases:
            with self.subTest(label=label):
                result = light_pipeline.annotate(*args, **kwargs)
                self.assertIsInstance(result, expected_type, f"{label} should return {expected_type.__name__}")
                for item in result:
                    self.assertIsInstance(item, dict)

        invalid_cases = [
            ("annotate(ids_only)", (self.ids,)),
            ("annotate(ids, str)", (self.ids, "wrong_type")),
            ("annotate(ids, list[int])", (self.ids, [1, 2])),
        ]

        for label, args in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaises(TypeError, msg=f"{label} should raise TypeError"):
                    light_pipeline.annotate(*args)


@pytest.mark.fast
class LightPipelineFullAnnotateIdsTextsInputTest(LightPipelineIdsTextsBase):
    """Tests fullAnnotate(ids, texts) and keyword equivalents."""

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        valid_cases = [
            ("fullAnnotate(ids, texts)", (self.ids, self.texts), {}, list),
            ("fullAnnotate(target=ids, optional_target=texts)",
             (), {"target": self.ids, "optional_target": self.texts}, list),
        ]

        for label, args, kwargs, expected_type in valid_cases:
            with self.subTest(label=label):
                result = light_pipeline.fullAnnotate(*args, **kwargs)
                self.assertIsInstance(result, expected_type, f"{label} should return {expected_type.__name__}")
                for item in result:
                    self.assertIsInstance(item, dict)
                    for v in item.values():
                        if v:
                            self.assertTrue(
                                all(isinstance(a, Annotation) for a in v),
                                f"{label} expected list of Annotation objects"
                            )

        invalid_cases = [
            ("fullAnnotate(ids_only)", (self.ids,)),
            ("fullAnnotate(ids, str)", (self.ids, "wrong_type")),
            ("fullAnnotate(ids, list[int])", (self.ids, [1, 2])),
        ]

        for label, args in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaises(TypeError, msg=f"{label} should raise TypeError"):
                    light_pipeline.fullAnnotate(*args)


class LightPipelineQuestionAnsweringBase(unittest.TestCase):
    """Shared setup for (question, context) LightPipeline tests."""

    def setUp(self):
        self.spark = SparkSessionForTest.spark
        self.text = "This is a text input"
        self.textDataSet = self.spark.createDataFrame([[self.text]]).toDF("text")

        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        regex_tok = RegexTokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        pipeline = Pipeline().setStages([document_assembler, regex_tok])
        self.model = pipeline.fit(self.textDataSet)

        # QA-style data
        self.question = "Where was Barack Obama born?"
        self.context = "Barack Obama was born in Hawaii."
        self.questions = ["Where was Barack Obama born?", "Who founded John Snow Labs?"]
        self.contexts = [
            "Barack Obama was born in Hawaii.",
            "John Snow Labs was founded in Delaware."
        ]


@pytest.mark.slow
class LightPipelineAnnotateQuestionAnsweringInputTest(LightPipelineQuestionAnsweringBase):
    """Tests annotate(question, context) and variants."""

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        valid_cases = [
            ("annotate(q, c)", (self.question, self.context), {}, dict),
            ("annotate(target=q, optional_target=c)",
             (), {"target": self.question, "optional_target": self.context}, dict),
            ("annotate(list_q, list_c)", (self.questions, self.contexts), {}, list),
        ]

        for label, args, kwargs, expected_type in valid_cases:
            with self.subTest(label=label):
                result = light_pipeline.annotate(*args, **kwargs)
                self.assertIsInstance(result, expected_type, f"{label} should return {expected_type.__name__}")

        invalid_cases = [
            ("annotate(q, list_c)", (self.question, self.contexts)),
            ("annotate(list_q, c)", (self.questions, self.context)),
        ]

        for label, args in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaises(TypeError, msg=f"{label} should raise TypeError"):
                    light_pipeline.annotate(*args)


@pytest.mark.slow
class LightPipelineFullAnnotateQuestionAnsweringInputTest(LightPipelineQuestionAnsweringBase):
    """Tests fullAnnotate(question, context) and variants."""

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        valid_cases = [
            ("fullAnnotate(q, c)", (self.question, self.context), {}, list),
            ("fullAnnotate(target=q, optional_target=c)",
             (), {"target": self.question, "optional_target": self.context}, list),
            ("fullAnnotate(list_q, list_c)", (self.questions, self.contexts), {}, list),
        ]

        for label, args, kwargs, expected_type in valid_cases:
            with self.subTest(label=label):
                result = light_pipeline.fullAnnotate(*args, **kwargs)
                self.assertIsInstance(result, expected_type, f"{label} should return {expected_type.__name__}")
                for item in result:
                    self.assertIsInstance(item, dict)
                    for v in item.values():
                        if v:
                            self.assertTrue(
                                all(isinstance(a, Annotation) for a in v),
                                f"{label} expected list of Annotation objects"
                            )

        invalid_cases = [
            ("fullAnnotate(q, list_c)", (self.question, self.contexts)),
            ("fullAnnotate(list_q, c)", (self.questions, self.context)),
        ]

        for label, args in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaises(TypeError, msg=f"{label} should raise TypeError"):
                    light_pipeline.fullAnnotate(*args)


@pytest.mark.slow
class LightPipelineAudioInputTest(LightPipelineAudioBase):

    def runTest(self):
        light_pipeline = LightPipeline(self.model)

        valid_cases = [
            ("fullAnnotate(list[float])", (self.audio_data,), {}, list),
            ("fullAnnotate(list[list[float]])", ([self.audio_data, self.audio_data],), {}, list),
            ("fullAnnotate(target=list[float])", (), {"target": self.audio_data}, list),
        ]

        for label, args, kwargs, expected_type in valid_cases:
            with self.subTest(label=label):
                result = light_pipeline.fullAnnotate(*args, **kwargs)
                self.assertIsInstance(result, expected_type, f"{label} should return {expected_type.__name__}")

                # Each item should be a dict with AnnotationAudio + Annotation entries
                for item in result:
                    self.assertIsInstance(item, dict)

                    for key, v in item.items():
                        if not v:
                            continue

                        # AudioAssembler stage → AnnotationAudio
                        if key == "audio_assembler":
                            self.assertTrue(
                                all(isinstance(a, AnnotationAudio) for a in v),
                                f"{label}: expected AnnotationAudio objects in '{key}'"
                            )
                        else:
                            # Wav2Vec2ForCTC outputs → Annotation
                            self.assertTrue(
                                all(isinstance(a, Annotation) for a in v),
                                f"{label}: expected Annotation objects in '{key}'"
                            )

        invalid_cases = [
            ("fullAnnotate(mixed list)", ([1, "bad"],)),
            ("annotate(list[float])", (self.audio_data,)),  # annotate() not supported for audio
        ]

        for label, args in invalid_cases:
            with self.subTest(label=label):
                with self.assertRaises(TypeError, msg=f"{label} should raise TypeError"):
                    if label.startswith("annotate"):
                        light_pipeline.annotate(*args)
                    else:
                        light_pipeline.fullAnnotate(*args)



@pytest.mark.slow
class LightPipelineImageInputTest(LightPipelineImageSetUp):

    def setUp(self):
        super().setUp()
        self.single_image_path = os.path.join(self.images_path, os.listdir(self.images_path)[0])
        # and a batch (list of paths)
        self.batch_image_paths = [
            os.path.join(self.images_path, f)
            for f in os.listdir(self.images_path)[:2]
        ]

    def runTest(self):
        light_pipeline = LightPipeline(self.vit_model)

        valid_cases = [
            ("fullAnnotate(image_path:str)", (self.single_image_path,), {}, list),
            ("fullAnnotate(image_paths:list[str])", (self.batch_image_paths,), {}, list),
            ("fullAnnotate(target=image_path)", (), {"target": self.single_image_path}, list),
        ]

        for label, args, kwargs, expected_type in valid_cases:
            with self.subTest(label=label):
                result = light_pipeline.fullAnnotate(*args, **kwargs)
                self.assertIsInstance(result, expected_type, f"{label} should return {expected_type.__name__}")

                for item in result:
                    self.assertIsInstance(item, dict)
                    for key, v in item.items():
                        if not v:
                            continue

                        # The assembler stage → AnnotationImage
                        if key == "image_assembler":
                            self.assertTrue(
                                all(isinstance(a, AnnotationImage) for a in v),
                                f"{label}: expected AnnotationImage objects in '{key}'"
                            )
                        else:
                            # ViT classification output
                            self.assertTrue(
                                all(isinstance(a, Annotation) for a in v),
                                f"{label}: expected Annotation objects in '{key}'"
                            )

        invalid_cases = [
            ("fullAnnotate(list[int])", ([1, 2, 3],)),
            ("fullAnnotate(nonexistent path)", ("/path/to/nowhere.png",)),  # returns placeholder AnnotationImage
            ("annotate(image_path)", (self.single_image_path,)),  # annotate() not supported for images
        ]

        for label, args in invalid_cases:
            with self.subTest(label=label):
                if "nonexistent" in label:
                    # Expect a single result with an empty AnnotationImage and no classes
                    result = light_pipeline.fullAnnotate(*args)
                    self.assertIsInstance(result, list)
                    self.assertEqual(len(result), 1, f"{label} should return one placeholder result")

                    item = result[0]
                    self.assertIn("image_assembler", item)
                    self.assertIn("class", item)

                    # image_assembler has one empty AnnotationImage
                    self.assertEqual(len(item["image_assembler"]), 1)
                    self.assertTrue(
                        all(isinstance(a, AnnotationImage) for a in item["image_assembler"]),
                        f"{label}: expected AnnotationImage in 'image_assembler'"
                    )
                    # class output should be empty
                    self.assertEqual(item["class"], [], f"{label}: expected empty 'class' list for invalid image path")

                elif label.startswith("annotate"):
                    with self.assertRaises(TypeError, msg=f"{label} should raise TypeError"):
                        light_pipeline.annotate(*args)
                else:
                    with self.assertRaises(TypeError, msg=f"{label} should raise TypeError"):
                        light_pipeline.fullAnnotate(*args)
