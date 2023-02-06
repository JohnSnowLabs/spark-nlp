/*
 * Copyright 2017-2023 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForQuestionAnswering
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.scalatest.flatspec.AnyFlatSpec

class ZeroShotNerModelTest extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  "ZeroShotNerModel" should "load a RoBertaForQuestionAnswering instance via pretrained" taggedAs SlowTest in {
    ZeroShotNerModel
      .pretrained("roberta_base_qa_squad2", "en", "public/models")
      .isInstanceOf[ZeroShotNerModel]
  }

  "ZeroShotNer" should "download a RoBertaForQuestionAnswering and save it as a ZeroShotNerModel" taggedAs SlowTest in {

    RoBertaForQuestionAnswering
      .pretrained()
      .write
      .overwrite
      .save("./tmp_roberta_for_qa")

    val loadedZeroShotNerModel = ZeroShotNerModel
      .load("./tmp_roberta_for_qa")
      .setCaseSensitive(true)
      .setPredictionThreshold(0.1f)

    loadedZeroShotNerModel.write.overwrite
      .save("./tmp_roberta_for_qa_zero_ner")

  }

  "ZeroShotRobertaNer" should "run zero shot NER and check the number of entities returned" taggedAs SlowTest in {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = SentenceDetectorDLModel
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val zeroShotNer = ZeroShotNerModel
      .pretrained("roberta_base_qa_squad2")
      .setEntityDefinitions(
        Map(
          "NAME" -> Array("What is his name?", "What is my name?"),
          "CITY" -> Array("Which city?", "Which is the city?"),
          "SOMETHING_ELSE" -> Array("What is her name?")))
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("zero_shot_ner")
      .setIgnoreEntities(Array("SOMETHING_ELSE"))

    val nerConverter = new NerConverter()
      .setInputCols(Array("sentence", "token", "zero_shot_ner"))
      .setOutputCol("ner_chunks")

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentenceDetector, tokenizer, zeroShotNer, nerConverter))

    val data = Seq(
      (
        "Hellen works in London, Paris and Berlin. My name is Clara Johnson, I live in New York and my sister Hellen lives in Paris.",
        6),
      ("John is a man who works in London, London and London.", 4)).toDF("text", "nEntities")

    val results = pipeline.fit(data).transform(data).cache()

    results
      .selectExpr("document", "explode(zero_shot_ner) AS entity")
      .select(
        col("document.result").getItem(0).alias("document"),
        col("entity.result"),
        col("entity.metadata.word"),
        col("entity.metadata.sentence"),
        col("entity.begin"),
        col("entity.end"),
        col("entity.metadata.confidence"),
        col("entity.metadata.question"))
      .show(truncate = false)

    results
      .selectExpr("size(ner_chunks)", "nEntities")
      .collect()
      .map(row => equals(row.get(0).asInstanceOf[Int], row.get(1).asInstanceOf[Int]))

    results.select("zero_shot_ner.result").show(1, false)
    results.select("ner_chunks.result").show(1, false)

    println(zeroShotNer.getEntityDefinitionsStr.mkString("Array(", ", ", ")"))
    println(zeroShotNer.getIgnoreEntities.mkString("Array(", ", ", ")"))
    println(zeroShotNer.getEntities.mkString("Array(", ", ", ")"))
  }
}
