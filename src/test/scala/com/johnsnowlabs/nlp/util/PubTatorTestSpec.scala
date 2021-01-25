package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.training.PubTator
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util._
import org.scalatest._

import java.io.File
import scala.reflect.io.Directory


class PubTatorTestSpec extends FlatSpec{

  "PubTator.readDataset" should "create conll-friendly dataframe" taggedAs FastTest in {

    //remove file if it's already there
    val directory = new Directory(new File("./pubtator-conll-test"))
    directory.deleteRecursively()
    val df = PubTator.readDataset(ResourceHelper.spark, "./src/test/resources/corpus_pubtator_sample.txt")
    CoNLLGenerator.exportConllFiles(df, "pubtator-conll-test")
    directory.deleteRecursively()

  }

}
