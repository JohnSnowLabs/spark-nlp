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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{LocalTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec
import org.apache.spark.sql.functions.{col, size}

class MxbaiEmbeddingsTestSpec extends AnyFlatSpec {

  "Mxbai Embeddings" should "run end to end pipeline test" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val testData = Seq(
      "This is an example sentence for testing embeddings.",
      "Each sentence is converted into a numerical representation.",
      "Machine learning models use these embeddings for various tasks.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetectorDL = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "en")
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val embeddings = MxbaiEmbeddings
      .pretrained()
      .setInputCols("sentences")
      .setOutputCol("Mxbai")

    val pipeline = new Pipeline().setStages(Array(document, sentenceDetectorDL, embeddings))

    val pipelineModel = pipeline.fit(testData)
    val transformed = pipelineModel.transform(testData)

    transformed.select("text", "Mxbai.embeddings").show()

    assert(transformed.columns.contains("Mxbai"), "because pipeline should output Mxbai column")

    val embeddingsDF = transformed.withColumn("embeddings", col("Mxbai.embeddings").getItem(0))
    val sizesArray: Array[Int] = embeddingsDF
      .select(size(col("embeddings")).as("size"))
      .collect()
      .map(row => row.getAs[Int]("size"))

    assert(
      sizesArray.forall(_ > 0),
      s"because all embeddings should have size greater than 0: " +
        s"\nsmallest size was ${sizesArray.min}")

    assert(
      sizesArray.forall(_ == sizesArray.head),
      s"because all embeddings should have the same size: " +
        s"\nsizes were ${sizesArray.mkString(", ")}")

    assert(
      transformed.count() === testData.count(),
      s"because output should have same number of rows as input: " +
        s"\nresult was ${transformed.count()} \nexpected is: ${testData.count()}")
  }

  "Mxbai Embeddings" should "correctly embed multiple sentences" taggedAs LocalTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq("This is an example sentence", "Each sentence is converted")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = MxbaiEmbeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("Mxbai")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(ddd).transform(ddd)
    pipelineDF.select("Mxbai.embeddings").show(truncate = false)
    val pipelineModel = pipeline.fit(ddd)
    pipelineModel.stages.last
      .asInstanceOf[MxbaiEmbeddings]
      .write
      .overwrite()
      .save("./tmp_forsequence_model")
  }

  it should "have embeddings of the same size" taggedAs LocalTest in {
    import ResourceHelper.spark.implicits._
    val testDf = Seq(
      "I like apples",
      "I like bananas \\n and other things \\n like icream \\n and cats",
      "I like rockets")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = MxbaiEmbeddings
      .pretrained()
      .setInputCols("document")
      .setOutputCol("Mxbai")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineModel = pipeline.fit(testDf)
    val pipelineDF = pipelineModel.transform(testDf)

    val embeddingsDF = pipelineDF.withColumn("embeddings", col("Mxbai.embeddings").getItem(0))
    val sizesArray: Array[Int] = embeddingsDF
      .select(size(col("embeddings")).as("size"))
      .collect()
      .map(row => row.getAs[Int]("size"))

    assert(sizesArray.forall(_ == sizesArray.head))
  }

  it should "work with sentences" taggedAs LocalTest in {
    import ResourceHelper.spark.implicits._
    val testData = "I really enjoy my job. This is amazing"
    val testDf = Seq(testData).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetectorDL = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "en")
      .setInputCols(Array("document"))
      .setOutputCol("sentences")

    val embeddings = MxbaiEmbeddings
      .pretrained()
      .setInputCols("sentences")
      .setOutputCol("Mxbai")

    val pipeline = new Pipeline().setStages(Array(document, sentenceDetectorDL, embeddings))

    val pipelineModel = pipeline.fit(testDf)
    val pipelineDF = pipelineModel.transform(testDf)

    pipelineDF.select("Mxbai.embeddings").show(false)
  }

  it should "not return empty embeddings" taggedAs LocalTest in {
    import ResourceHelper.spark.implicits._
    val interests = Seq(
      "I like music",
      "I like movies",
      "I like books",
      "I like sports",
      "I like travel",
      "I like food",
      "I like games",
      "I like art",
      "I like nature",
      "I like science",
      "I like technology",
      "I like history",
      "I like fashion",
      "I like cars",
      "I like animals",
      "I like gardening")
    val testDf = interests.toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = MxbaiEmbeddings
      .pretrained()
      .setInputCols("document")
      .setOutputCol("Mxbai")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineModel = pipeline.fit(testDf)
    val pipelineDF = pipelineModel.transform(testDf)

    val embeddingsDF = pipelineDF.withColumn("embeddings", col("Mxbai.embeddings").getItem(0))
    val sizesArray: Array[Int] = embeddingsDF
      .select(size(col("embeddings")).as("size"))
      .collect()
      .map(row => row.getAs[Int]("size"))

    assert(sizesArray.forall(_ > 0))
  }

}
