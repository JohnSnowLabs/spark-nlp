package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.annotator.{NerDLModel, PerceptronModel, SentenceDetector}
import org.scalatest._
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{ContentProvider, DataBuilder, DocumentAssembler, Finisher}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.explode
import com.johnsnowlabs.util._
import org.apache.spark.ml.Pipeline

import scala.reflect.io.Directory
import java.io.File

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

import scala.io.Source

class CoNLLGeneratorTestSpec extends FlatSpec{
  ResourceHelper.spark
  import ResourceHelper.spark.implicits._ //for toDS and toDF

  val preModel = PretrainedPipeline("explain_document_ml", lang="en").model

  val finisher = new Finisher()
    .setInputCols("token", "pos")
    .setIncludeMetadata(true)

  val ourPipelineModel = new Pipeline()
    .setStages(Array(preModel, finisher))
    .fit(Seq("").toDF("text"))

  val testing = Seq(
    (1, "Google is a famous company"),
    (2, "Peter Parker is a super heroe"))
    .toDS.toDF( "_id", "text")

  val result = ourPipelineModel.transform(testing)


  //TODO: read this from a file?
  //this is what the output should be
  val testText = """"" "" "" ""
      |-DOCSTART- -X- -X- O
      |"" "" "" ""
      |"" "" "" ""
      |Google NNP NNP O
      |is VBZ VBZ O
      |a DT DT O
      |famous JJ JJ O
      |company NN NN O
      |"" "" "" ""
      |-DOCSTART- -X- -X- O
      |"" "" "" ""
      |"" "" "" ""
      |Peter NNP NNP O
      |Parker NNP NNP O
      |is VBZ VBZ O
      |a DT DT O
      |super JJ JJ O
      |heroe NN NN O""".stripMargin.replace("\n", "" )

  "The dataframe, pipelinemodel, outputpath generator" should "make the right file" in {
    CoNLLGenerator.exportConllFiles(result, ourPipelineModel, "./testcsv")

    //read csv, check if it's equal
    def getPath(dir: String): String = {
      val file = new File(dir)
      file.listFiles.filter(_.isFile)
        .filter(_.getName.endsWith(".csv"))
        .map(_.getPath).head
    }

    val filePath = getPath("./testcsv/")

    val fileContents = Source.fromFile(filePath).getLines.mkString

    //delete the csv and folder we just made
    val directory = new Directory(new File("./testcsv"))
    directory.deleteRecursively()

    assert(fileContents==testText)
  }

  "The dataframe, outputpath generator" should "make the right file" in {
    CoNLLGenerator.exportConllFiles(result, "./testcsv")

    //read csv, check if it's equal
    def getPath(dir: String): String = {
      val file = new File(dir)
      file.listFiles.filter(_.isFile)
        .filter(_.getName.endsWith(".csv"))
        .map(_.getPath).head
    }

    val filePath = getPath("./testcsv/")

    val fileContents = Source.fromFile(filePath).getLines.mkString

    //delete the csv and folder we just made
    val directory = new Directory(new File("./testcsv"))
    directory.deleteRecursively()

    assert(fileContents==testText)
  }
}
