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

package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.Finisher
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util._
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File
import scala.io.Source
import scala.reflect.io.Directory

class CoNLLGeneratorTestSpec extends AnyFlatSpec {
  ResourceHelper.spark
  import ResourceHelper.spark.implicits._ // for toDS and toDF

  "The (dataframe, pipelinemodel, outputpath) generator" should "make the right file" taggedAs SlowTest in {
    val preModel = PretrainedPipeline("explain_document_dl", lang = "en").model

    val finisherNoNER = new Finisher()
      .setInputCols("token", "pos")
      .setIncludeMetadata(true)

    val ourPipelineModelNoNER = new Pipeline()
      .setStages(Array(preModel, finisherNoNER))
      .fit(Seq("").toDF("text"))

    val testing = Seq(
      (1, "Google is a famous company"),
      (2, "Peter Parker is a super heroe")).toDS.toDF("_id", "text")

    // TODO: read this from a file?
    // this is what the generator output should be
    val testText =
      "   -DOCSTART- -X- -X- O      Google NNP NNP Ois VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP OParker NNP NNP Ois VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"
    val testNERText =
      "   -DOCSTART- -X- -X- O      Google NNP NNP B-ORGis VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP B-PERParker NNP NNP I-PERis VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"

    // remove file if it's already there
    val directory = new Directory(new File("./testcsv"))
    directory.deleteRecursively()

    CoNLLGenerator.exportConllFiles(testing, ourPipelineModelNoNER, "./testcsv")

    // read csv, check if it's equal
    def getPath(dir: String): String = {
      val file = new File(dir)
      file.listFiles
        .filter(_.isFile)
        .filter(_.getName.endsWith(".csv"))
        .map(_.getPath)
        .head
    }
    val filePath = getPath("./testcsv/")
    val fileContents = Source.fromFile(filePath).getLines.mkString
    directory.deleteRecursively()

    assert(fileContents == testText)
  }

  "The (dataframe, outputpath) generator" should "make the right file" taggedAs SlowTest in {
    val preModel = PretrainedPipeline("explain_document_dl", lang = "en").model

    val finisherNoNER = new Finisher()
      .setInputCols("token", "pos")
      .setIncludeMetadata(true)

    val ourPipelineModelNoNER = new Pipeline()
      .setStages(Array(preModel, finisherNoNER))
      .fit(Seq("").toDF("text"))

    val testing = Seq(
      (1, "Google is a famous company"),
      (2, "Peter Parker is a super heroe")).toDS.toDF("_id", "text")

    val resultNoNER = ourPipelineModelNoNER.transform(testing)

    // TODO: read this from a file?
    // this is what the generator output should be
    val testText =
      "   -DOCSTART- -X- -X- O      Google NNP NNP Ois VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP OParker NNP NNP Ois VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"
    val testNERText =
      "   -DOCSTART- -X- -X- O      Google NNP NNP B-ORGis VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP B-PERParker NNP NNP I-PERis VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"

    // remove file if it's already there
    val directory = new Directory(new File("./testcsv"))
    directory.deleteRecursively()

    CoNLLGenerator.exportConllFiles(resultNoNER, "./testcsv")

    // read csv, check if it's equal
    def getPath(dir: String): String = {
      val file = new File(dir)
      file.listFiles
        .filter(_.isFile)
        .filter(_.getName.endsWith(".csv"))
        .map(_.getPath)
        .head
    }
    val filePath = getPath("./testcsv/")
    val fileContents = Source.fromFile(filePath).getLines.mkString
    directory.deleteRecursively()

    assert(fileContents == testText)
  }

  "The generator" should "make the right file with ners when appropriate" taggedAs SlowTest in {
    val preModel = PretrainedPipeline("explain_document_dl", lang = "en").model

    val testing = Seq(
      (1, "Google is a famous company"),
      (2, "Peter Parker is a super heroe")).toDS.toDF("_id", "text")

    val finisherWithNER = new Finisher()
      .setInputCols("token", "pos", "ner")
      .setIncludeMetadata(true)

    val ourPipelineModelWithNER = new Pipeline()
      .setStages(Array(preModel, finisherWithNER))
      .fit(Seq("").toDF("text"))

    val resultWithNER = ourPipelineModelWithNER.transform(testing)

    // TODO: read this from a file?
    // this is what the generator output should be
    val testText =
      "   -DOCSTART- -X- -X- O      Google NNP NNP Ois VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP OParker NNP NNP Ois VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"
    val testNERText =
      "   -DOCSTART- -X- -X- O      Google NNP NNP B-ORGis VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP B-PERParker NNP NNP I-PERis VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"

    // remove file if it's already there
    val directory = new Directory(new File("./testcsv"))
    directory.deleteRecursively()

    CoNLLGenerator.exportConllFiles(resultWithNER, "./testcsv")

    // read csv, check if it's equal
    def getPath(dir: String): String = {
      val file = new File(dir)
      file.listFiles
        .filter(_.isFile)
        .filter(_.getName.endsWith(".csv"))
        .map(_.getPath)
        .head
    }
    val filePath = getPath("./testcsv/")
    val fileContents = Source.fromFile(filePath).getLines.mkString
    directory.deleteRecursively()

    assert(fileContents == testNERText)
  }

  "The generator" should "work even if token metadata has non-ints" in {
    val df = ResourceHelper.spark.read.load(
      "src/test/resources/conllgenerator/conllgenerator_nonint_token_metadata.parquet")

    CoNLLGenerator.exportConllFiles(df, "./tmp_noninttokens")
  }
}
