package com.johnsnowlabs.nlp.annotators.ner.dl

import org.apache.spark.{SparkConf, SparkContext}

class GenerateGraph {

  def createGraph(scriptPath: String): Unit = {
    import sys.process._

    val sparkConf = new SparkConf().setAppName("ScalaPython").setMaster("local")
    val sparkContext = new SparkContext(sparkConf)
    val data = List("john")
    val dataRDD = sparkContext.makeRDD(data)
    val pipeRDD = dataRDD.pipe(scriptPath)
    pipeRDD.foreach(println)
    val stdout = new StringBuilder
    val stderr = new StringBuilder
    val status = "ls -al FRED" ! ProcessLogger(stdout append _, stderr append _)
    println("This println is from Scala Yei!! " + status)
  }

}
