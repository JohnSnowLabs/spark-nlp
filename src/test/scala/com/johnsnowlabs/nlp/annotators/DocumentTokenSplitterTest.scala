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
package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

class DocumentTokenSplitterTest extends AnyFlatSpec {

  val spark = ResourceHelper.spark

  import spark.implicits._

  val text =
    "All emotions, and that\none particularly, were abhorrent to his cold, precise but\nadmirably balanced mind.\n\n" +
      "He was, I take it, the most perfect\nreasoning and observing machine that the world has seen."

  val textDf = Seq(text).toDF("text")
  val documentAssembler = new DocumentAssembler().setInputCol("text")
  val textDocument: DataFrame = documentAssembler.transform(textDf)

  behavior of "DocumentTokenTextSplitter"

  it should "split by number of tokens" taggedAs FastTest in {
    val numTokens = 3
    val tokenTextSplitter =
      new DocumentTokenSplitter()
        .setInputCols("document")
        .setOutputCol("splits")
        .setNumTokens(numTokens)

    val splitDF = tokenTextSplitter.transform(textDocument)
    val result = Annotation.collect(splitDF, "splits").head

    result.foreach(annotation => assert(annotation.metadata("numTokens").toInt == numTokens))
  }

  it should "split tokens with overlap" taggedAs FastTest in {
    val numTokens = 3
    val tokenTextSplitter =
      new DocumentTokenSplitter()
        .setInputCols("document")
        .setOutputCol("splits")
        .setNumTokens(numTokens)
        .setTokenOverlap(1)

    val splitDF = tokenTextSplitter.transform(textDocument)
    val result = Annotation.collect(splitDF, "splits").head

    result.zipWithIndex.foreach { case (annotation, i) =>
      if (i < result.length - 1) // Last document is shorter
        assert(annotation.metadata("numTokens").toInt == numTokens)
    }
  }

  it should "be serializable" taggedAs FastTest in {
    val numTokens = 3
    val textSplitter = new DocumentTokenSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setNumTokens(numTokens)
      .setTokenOverlap(1)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, textSplitter))
    val pipelineModel = pipeline.fit(textDf)

    pipelineModel.stages.last
      .asInstanceOf[DocumentTokenSplitter]
      .write
      .overwrite()
      .save("./tmp_textSplitter")

    val loadedTextSplitModel = DocumentTokenSplitter.load("tmp_textSplitter")

    loadedTextSplitModel.transform(textDocument).select("splits").show(truncate = false)
  }

  it should "be exportable to pipeline" taggedAs FastTest in {
    val numTokens = 3
    val textSplitter = new DocumentTokenSplitter()
      .setInputCols("document")
      .setOutputCol("splits")
      .setNumTokens(numTokens)
      .setTokenOverlap(1)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, textSplitter))
    pipeline.write.overwrite().save("tmp_textsplitter_pipe")

    val loadedPipelineModel = Pipeline.load("tmp_textsplitter_pipe")

    loadedPipelineModel.fit(textDf).transform(textDf).select("splits").show()
  }

}
