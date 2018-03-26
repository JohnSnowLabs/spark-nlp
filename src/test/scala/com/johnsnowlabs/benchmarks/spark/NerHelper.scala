package com.johnsnowlabs.benchmarks.spark

import java.io.{BufferedWriter, File, FileWriter}

import com.johnsnowlabs.nlp.Annotation


object NerHelper {

  /**
    * Print top n Named Entity annotations
    */
  def print(annotations: Seq[Annotation], n: Int): Unit = {
    for (a <- annotations.take(n)) {
      System.out.println(s"${a.begin}, ${a.end}, ${a.result}, ${a.metadata("text")}")
    }
  }

  /**
    * Saves ner results to csv file
    * @param annotations
    * @param file
    */
  def saveNerSpanTags(annotations: Array[Array[Annotation]], file: String): Unit = {
    val bw = new BufferedWriter(new FileWriter(new File(file)))

    bw.write(s"start\tend\ttag\ttext\n")
    for (i <- 0 until annotations.length) {
      for (a <- annotations(i))
        bw.write(s"${a.begin}\t${a.end}\t${a.result}\t${a.metadata("text").replace("\n", " ")}\n")
    }
    bw.close()
  }
}
