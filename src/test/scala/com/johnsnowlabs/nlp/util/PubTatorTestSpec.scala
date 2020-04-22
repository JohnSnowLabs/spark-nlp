package com.johnsnowlabs.nlp.util

import java.io.File

import com.johnsnowlabs.nlp.Finisher
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util._
import org.apache.spark.ml.Pipeline
import org.scalatest._
import com.johnsnowlabs.nlp.training.PubTator

import scala.io.Source
import scala.reflect.io.Directory


class PubTatorTestSpec extends FlatSpec{

  "PubTator.readDataset" should "make the right file" in {
    ResourceHelper.spark
    import ResourceHelper.spark.implicits._ //for toDS and toDF
    //TODO: change to absolute path, add corpus to repo
    PubTator().readDataset(ResourceHelper.spark, "/home/brian/jsl-repos/corpus_pubtator.txt")


  assert(true)
  }

}
