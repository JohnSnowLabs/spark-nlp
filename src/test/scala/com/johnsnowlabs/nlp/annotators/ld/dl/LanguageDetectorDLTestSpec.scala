/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.ld.dl

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.{DocumentAssembler, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class LanguageDetectorDLTestSpec extends AnyFlatSpec {

  val spark = ResourceHelper.spark
  import spark.implicits._

  "LanguageDetectorDL" should "correctly load pretrained model" taggedAs SlowTest in {

    val smallCorpus = spark.read
      .option("header", true)
      .option("delimiter", "|")
      .csv("src/test/resources/language-detector/multilingual_sample.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val languageDetector = LanguageDetectorDL
      .pretrained()
      .setInputCols("sentence")
      .setOutputCol("language")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, languageDetector))

    pipeline.fit(smallCorpus).write.overwrite().save("./tmp_ld_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_ld_pipeline")
    pipelineModel.transform(smallCorpus).select("language.result", "lang").show(2, false)

  }

  private implicit val formats: DefaultFormats.type = DefaultFormats

  private lazy val glotlid = LanguageDetectorDL.pretrained("ld_glotlid_v4", "xx")

  private def configured(
      threshold: Float = 0.7f,
      scriptPooling: Boolean = true,
      minLetters: Int = 20,
      topK: Int = 5,
      coalesceSentences: Boolean = false): LanguageDetectorDL =
    glotlid
      .setInputCols("document")
      .setOutputCol("language")
      .setThreshold(threshold)
      .setScriptPooling(scriptPooling)
      .setMinLetters(minLetters)
      .setTopK(topK)
      .setCoalesceSentences(coalesceSentences)

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

  private def detect(languageDetector: LanguageDetectorDL, texts: Seq[String]): Seq[Annotation] = {
    val data = texts.zipWithIndex.toDF("text", "id")
    val result = new Pipeline()
      .setStages(Array(documentAssembler, languageDetector))
      .fit(data)
      .transform(data)
      .orderBy("id")
    Annotation.collect(result, "language").map(_.head)
  }

  // fastText predictions for FLORES-200, medical and edge-case sentences
  private lazy val reference = {
    val source =
      Source.fromFile("src/test/resources/language-detector/glotlid/predictions.json", "UTF-8")
    val json =
      try parse(source.mkString)
      finally source.close()
    (json \ "sentences").children.filter(s => (s \ "text").extract[String].nonEmpty)
  }

  private val texts = Seq(
    "The patient was admitted with chest pain and shortness of breath.",
    "患者因胸痛和呼吸困难三天入院。",
    "El paciente ingresó por dolor torácico y dificultad para respirar.",
    "Bệnh nhân nhập viện vì đau ngực.",
    "Ang pasyente ay na-admit dahil sa pananakit ng dibdib.",
    "def mean(xs): return sum(xs) / len(xs)")

  private val labels = Seq("eng_Latn", "cmn_Hani", "spa_Latn", "vie_Latn", "tgl_Latn", "zxx_Code")

  "LanguageDetectorDL with GlotLID v4" should "load with the pretrained settings" taggedAs SlowTest in {
    assert(glotlid.getEngine == "onnx")
    assert(glotlid.getThreshold == 0.7f)
    assert(glotlid.getThresholdLabel == "unk")
    assert(glotlid.getScriptPooling)
    assert(glotlid.getMinLetters == 20)
    assert(glotlid.getTopK == 5)
    assert(!glotlid.getCoalesceSentences)
    assert(glotlid.getLanguage.length == 2060)
  }

  it should "detect the language of each text" taggedAs SlowTest in {
    val annotations = detect(configured(), texts)
    assert(annotations.map(_.result) == labels)
    assert(annotations.forall(_.metadata("confidence").toFloat >= 0.7f))
  }

  it should "give fastText's top label and scores" taggedAs SlowTest in {
    val annotations = detect(
      configured(threshold = 0f, scriptPooling = false, minLetters = 0, topK = 0),
      reference.map(s => (s \ "text").extract[String]))

    reference.zip(annotations).foreach { case (s, annotation) =>
      val top = (s \ "top").children.map(e => (e(0).extract[String], e(1).extract[Double]))
      assert(annotation.result == top.head._1, (s \ "name").extract[String])
      top.foreach { case (label, p) =>
        assert(math.abs(annotation.metadata(label).toDouble - p) <= 1e-4, label)
      }
    }
  }

  it should "pool the scores of a non-Latin script" taggedAs SlowTest in {
    val text = Seq("血液檢查顯示肌鈣蛋白升高,胸部X光片符合肺水腫表現。")
    assert(detect(configured(scriptPooling = false), text).head.result == "unk")
    assert(detect(configured(scriptPooling = true), text).head.result == "yue_Hani")
  }

  it should "give thresholdLabel to short Latin-script text" taggedAs SlowTest in {
    val text = Seq("No oral lesions.")
    assert(detect(configured(minLetters = 0), text).head.result == "cat_Latn")
    assert(detect(configured(minLetters = 20), text).head.result == "unk")
  }

  it should "keep topK scores in the metadata" taggedAs SlowTest in {
    val metadata = detect(configured(topK = 2), texts.take(1)).head.metadata
    assert(metadata.size == 4)
    assert(metadata.contains("eng_Latn"))
  }

  it should "average the sentence scores when coalesceSentences is set" taggedAs SlowTest in {
    val data = Seq(
      "The patient was admitted with chest pain. El paciente ingresó por dolor torácico.").toDF(
      "text")
    def run(coalesce: Boolean): Seq[Annotation] = {
      val languageDetector =
        configured(threshold = 0f, topK = 0, coalesceSentences = coalesce).setInputCols("sentence")
      val result = new Pipeline()
        .setStages(Array(documentAssembler, sentenceDetector, languageDetector))
        .fit(data)
        .transform(data)
      Annotation.collect(result, "language").head
    }

    val sentences = run(coalesce = false)
    val document = run(coalesce = true)
    assert(sentences.map(_.result) == Seq("eng_Latn", "spa_Latn"))
    assert(document.size == 1)
    Seq("eng_Latn", "spa_Latn").foreach { label =>
      val average = sentences.map(_.metadata(label).toDouble).sum / 2
      assert(math.abs(document.head.metadata(label).toDouble - average) <= 1e-6, label)
    }
  }

  it should "give the sentences of a long document the same scores as one by one" taggedAs SlowTest in {
    val data = Seq(reference.map(s => (s \ "text").extract[String]).mkString(" ")).toDF("text")
    val result = new Pipeline()
      .setStages(Array(documentAssembler, sentenceDetector, configured().setInputCols("sentence")))
      .fit(data)
      .transform(data)
    val inDocument = Annotation.collect(result, "language").head.toSeq
    val sentences = Annotation.collect(result, "sentence").head.map(_.result).toSeq
    val oneByOne = detect(configured(), sentences)

    assert(inDocument.size == sentences.size)
    inDocument.zip(oneByOne).foreach { case (a, b) =>
      assert(a.result == b.result)
      assert(
        math.abs(a.metadata("confidence").toDouble - b.metadata("confidence").toDouble) <= 1e-6)
    }
  }

  it should "be saved and loaded" taggedAs SlowTest in {
    val data = texts.toDF("text")
    new Pipeline()
      .setStages(Array(documentAssembler, configured()))
      .fit(data)
      .write
      .overwrite()
      .save("./tmp_ld_glotlid_pipeline")
    val pipelineModel = PipelineModel.load("./tmp_ld_glotlid_pipeline")

    val result = pipelineModel.transform(data)
    result.select("text", "language.result").show(truncate = false)
    assert(Annotation.collect(result, "language").map(_.head.result).toSeq == labels)
  }

  it should "annotate in a LightPipeline" taggedAs SlowTest in {
    val pipelineModel = new Pipeline()
      .setStages(Array(documentAssembler, configured()))
      .fit(Seq("").toDF("text"))

    val result = new LightPipeline(pipelineModel).annotate(texts.head)
    assert(result("language") == Seq("eng_Latn"))
  }

  "LanguageDetectorDL with GlotLID v4 int8" should "detect the language of each text" taggedAs SlowTest in {
    val languageDetector = LanguageDetectorDL
      .pretrained("ld_glotlid_v4_int8", "xx")
      .setInputCols("document")
      .setOutputCol("language")
    assert(detect(languageDetector, texts).map(_.result) == labels)
  }
}
