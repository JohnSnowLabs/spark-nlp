package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.Finisher
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util._
import org.apache.spark.ml.Pipeline
import org.scalatest._

import java.io.File
import scala.io.Source
import scala.reflect.io.Directory

class CoNLLGeneratorTestSpec extends FlatSpec{
  ResourceHelper.spark
  import ResourceHelper.spark.implicits._ //for toDS and toDF

  "The (dataframe, pipelinemodel, outputpath) generator" should "make the right file" taggedAs SlowTest in {
    val preModel = PretrainedPipeline("explain_document_dl", lang="en").model

    val finisherNoNER = new Finisher()
      .setInputCols("token", "pos")
      .setIncludeMetadata(true)

    val ourPipelineModelNoNER = new Pipeline()
      .setStages(Array(preModel, finisherNoNER))
      .fit(Seq("").toDF("text"))

    val testing = Seq(
      (1, "Google is a famous company"),
      (2, "Peter Parker is a super heroe"))
      .toDS.toDF( "_id", "text")

    val resultNoNER = ourPipelineModelNoNER.transform(testing)


    val ner = NerDLModel.pretrained("ner_dl")
      .setInputCols(Array("document", "token", "embeddings"))
      .setOutputCol("ner")

    val finisherWithNER = new Finisher()
      .setInputCols("token", "pos", "ner")
      .setIncludeMetadata(true)

    val ourPipelineModelWithNER = new Pipeline()
      .setStages(Array(preModel, ner, finisherWithNER))
      .fit(Seq("").toDF("text"))

    val resultWithNER = ourPipelineModelWithNER.transform(testing)

    //TODO: read this from a file?
    //this is what the generator output should be
    val testText = "   -DOCSTART- -X- -X- O      Google NNP NNP Ois VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP OParker NNP NNP Ois VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"
    val testNERText = "   -DOCSTART- -X- -X- O      Google NNP NNP B-ORGis VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP B-PERParker NNP NNP I-PERis VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"

    //remove file if it's already there
    val directory = new Directory(new File("./testcsv"))
    directory.deleteRecursively()

    CoNLLGenerator.exportConllFiles(testing, ourPipelineModelNoNER, "./testcsv")

    //read csv, check if it's equal
    def getPath(dir: String): String = {
      val file = new File(dir)
      file.listFiles.filter(_.isFile)
        .filter(_.getName.endsWith(".csv"))
        .map(_.getPath).head
    }
    val filePath = getPath("./testcsv/")
    val fileContents = Source.fromFile(filePath).getLines.mkString
    directory.deleteRecursively()

    assert(fileContents==testText)
  }


  "The (dataframe, outputpath) generator" should "make the right file" taggedAs SlowTest in {
    val preModel = PretrainedPipeline("explain_document_dl", lang="en").model

    val finisherNoNER = new Finisher()
      .setInputCols("token", "pos")
      .setIncludeMetadata(true)

    val ourPipelineModelNoNER = new Pipeline()
      .setStages(Array(preModel, finisherNoNER))
      .fit(Seq("").toDF("text"))

    val testing = Seq(
      (1, "Google is a famous company"),
      (2, "Peter Parker is a super heroe"))
      .toDS.toDF( "_id", "text")

    val resultNoNER = ourPipelineModelNoNER.transform(testing)


    val ner = NerDLModel.pretrained("ner_dl")
      .setInputCols(Array("document", "token", "embeddings"))
      .setOutputCol("ner")

    val finisherWithNER = new Finisher()
      .setInputCols("token", "pos", "ner")
      .setIncludeMetadata(true)

    val ourPipelineModelWithNER = new Pipeline()
      .setStages(Array(preModel, ner, finisherWithNER))
      .fit(Seq("").toDF("text"))

    val resultWithNER = ourPipelineModelWithNER.transform(testing)

    //TODO: read this from a file?
    //this is what the generator output should be
    val testText = "   -DOCSTART- -X- -X- O      Google NNP NNP Ois VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP OParker NNP NNP Ois VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"
    val testNERText = "   -DOCSTART- -X- -X- O      Google NNP NNP B-ORGis VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP B-PERParker NNP NNP I-PERis VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"

    //remove file if it's already there
    val directory = new Directory(new File("./testcsv"))
    directory.deleteRecursively()

    CoNLLGenerator.exportConllFiles(resultNoNER, "./testcsv")

    //read csv, check if it's equal
    def getPath(dir: String): String = {
      val file = new File(dir)
      file.listFiles.filter(_.isFile)
        .filter(_.getName.endsWith(".csv"))
        .map(_.getPath).head
    }
    val filePath = getPath("./testcsv/")
    val fileContents = Source.fromFile(filePath).getLines.mkString
    directory.deleteRecursively()

    assert(fileContents==testText)
  }


  "The generator" should "make the right file with ners when appropriate" taggedAs SlowTest in {
    val preModel = PretrainedPipeline("explain_document_dl", lang="en").model

    val finisherNoNER = new Finisher()
      .setInputCols("token", "pos")
      .setIncludeMetadata(true)

    val ourPipelineModelNoNER = new Pipeline()
      .setStages(Array(preModel, finisherNoNER))
      .fit(Seq("").toDF("text"))

    val testing = Seq(
      (1, "Google is a famous company"),
      (2, "Peter Parker is a super heroe"))
      .toDS.toDF( "_id", "text")

    val resultNoNER = ourPipelineModelNoNER.transform(testing)


    val ner = NerDLModel.pretrained("ner_dl")
      .setInputCols(Array("document", "token", "embeddings"))
      .setOutputCol("ner")

    val finisherWithNER = new Finisher()
      .setInputCols("token", "pos", "ner")
      .setIncludeMetadata(true)

    val ourPipelineModelWithNER = new Pipeline()
      .setStages(Array(preModel, ner, finisherWithNER))
      .fit(Seq("").toDF("text"))

    val resultWithNER = ourPipelineModelWithNER.transform(testing)

    //TODO: read this from a file?
    //this is what the generator output should be
    val testText = "   -DOCSTART- -X- -X- O      Google NNP NNP Ois VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP OParker NNP NNP Ois VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"
    val testNERText = "   -DOCSTART- -X- -X- O      Google NNP NNP B-ORGis VBZ VBZ Oa DT DT Ofamous JJ JJ Ocompany NN NN O   -DOCSTART- -X- -X- O      Peter NNP NNP B-PERParker NNP NNP I-PERis VBZ VBZ Oa DT DT Osuper JJ JJ Oheroe NN NN O"

    //remove file if it's already there
    val directory = new Directory(new File("./testcsv"))
    directory.deleteRecursively()

    CoNLLGenerator.exportConllFiles(resultWithNER, "./testcsv")

    //read csv, check if it's equal
    def getPath(dir: String): String = {
      val file = new File(dir)
      file.listFiles.filter(_.isFile)
        .filter(_.getName.endsWith(".csv"))
        .map(_.getPath).head
    }
    val filePath = getPath("./testcsv/")
    val fileContents = Source.fromFile(filePath).getLines.mkString
    directory.deleteRecursively()

    assert(fileContents==testNERText)
  }

}