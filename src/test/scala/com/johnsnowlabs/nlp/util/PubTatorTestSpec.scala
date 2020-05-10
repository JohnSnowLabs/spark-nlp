package com.johnsnowlabs.nlp.util

import java.io.File

import com.johnsnowlabs.nlp.training.PubTator
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util._
import org.scalatest._

import scala.reflect.io.Directory


class PubTatorTestSpec extends FlatSpec{

  "PubTator.readDataset" should "create conll-friendly dataframe" in {

    //remove file if it's already there
    val directory = new Directory(new File("./pubtator-conll-test"))
    directory.deleteRecursively()
    val df = PubTator.readDataset(ResourceHelper.spark, "./src/test/resources/corpus_pubtator_sample.txt")
    CoNLLGenerator.exportConllFiles(df, "pubtator-conll-test")
    directory.deleteRecursively()

  }

}
