package com.johnsnowlabs.nlp.annotators.spell.context

trait OcrTestData {

  val correctFile : String
  val rawFile: String

  def loadDataSets() : (Array[Array[String]], Array[Array[String]]) = {

    val correct = scala.io.Source.fromFile(correctFile).getLines.map {line =>
      line.split(" ")}.toArray

    val raw = scala.io.Source.fromFile(rawFile).getLines.map {line =>
      line.split(" ")}.toArray

    correct.zip(raw).map { case (c, r) =>
        if (c.size != r.size)
          println(s"Difference in length\n ${c.mkString(" ")} \n ${r.mkString(" ")}")
    }

    (correct, raw)
  }

}
