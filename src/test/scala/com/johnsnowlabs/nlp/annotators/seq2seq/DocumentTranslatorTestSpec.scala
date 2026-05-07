/*
 * Copyright 2017-2025 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.reader.Reader2Doc
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class DocumentTranslatorTestSpec extends AnyFlatSpec with SparkSessionTest {

  val htmlDirectory = "src/test/resources/reader/html"
  val pdfDirectory = "src/test/resources/reader/pdf"
  val txtDirectory = "src/test/resources/reader/txt"

  "DocumentTranslator" should "translate an HTML document from English to French" taggedAs SlowTest in {

    val documentTranslator = DocumentTranslator
      .pretrained("opus_mt_en_fr", "xx")
      .setContentType("text/html")
      .setContentPath(s"$htmlDirectory/fake-html.html")
      .setOutputCol("translation")

    val pipeline = new Pipeline().setStages(Array(documentTranslator))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
    val translationResults = resultDf.select("translation.result").head().getSeq[String](0)
    assert(translationResults.nonEmpty)
  }

  it should "translate a PDF document from English to French" taggedAs SlowTest in {

    val documentTranslator = DocumentTranslator
      .pretrained("opus_mt_en_fr", "xx")
      .setContentType("application/pdf")
      .setContentPath(s"$pdfDirectory/pdf-title.pdf")
      .setOutputCol("translation")

    val pipeline = new Pipeline().setStages(Array(documentTranslator))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
    val translationResults = resultDf.select("translation.result").head().getSeq[String](0)
    assert(translationResults.nonEmpty)
  }

  it should "translate a plain-text document from English to French" taggedAs SlowTest in {

    val documentTranslator = DocumentTranslator
      .pretrained("opus_mt_en_fr", "xx")
      .setContentType("text/plain")
      .setContentPath(s"$txtDirectory/long-text.txt")
      .setOutputCol("translation")
      .setMaxInputLength(512)
      .setMaxOutputLength(512)
      .setChunkSize(450)

    val pipeline = new Pipeline().setStages(Array(documentTranslator))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    assert(resultDf.count() == 1)
    val translationResults = resultDf.select("translation.result").head().getSeq[String](0)
    resultDf.show(truncate = false)
    assert(translationResults.nonEmpty)
  }

  it should "detect and count sentences in a plain-text document using Reader2Doc and SentenceDetectorDL" taggedAs SlowTest in {

    val reader = new Reader2Doc()
      .setContentType("text/plain")
      .setContentPath(s"$txtDirectory/long-text.txt")
      .setOutputCol("document")
      .setOutputAsDocument(true)

    val sentenceDetector = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "en")
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val pipeline = new Pipeline().setStages(Array(reader, sentenceDetector))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    resultDf.select("fileName", "sentences").show(truncate = false)

    val sentenceCount = resultDf.select("sentences").head().getSeq[org.apache.spark.sql.Row](0).length
    println(s"[DocumentTranslatorTestSpec] Number of sentences detected: $sentenceCount")

    assert(sentenceCount > 0, "Expected at least one sentence to be detected")
  }
}
