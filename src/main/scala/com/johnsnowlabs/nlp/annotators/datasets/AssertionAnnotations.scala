package com.johnsnowlabs.nlp.annotators.datasets

import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.sql.Row

/**
  * Created by jose on 12/01/18.
  */
case class AssertionAnnotationAndText(text: String, target: String, label: String, start:Int, end:Int)
case class AssertionAnnotationWithLabel(label: String, start:Int, end:Int)
case class IndexedChunk(chunkBegin: Int, chunkEnd: Int)
object AssertionAnnotationWithLabel {

  def fromChunk(sentence: String, label: String, chunkContent: Seq[Row]): Seq[AssertionAnnotationWithLabel] = {
    val chunks = chunkContent.map { r => Annotation(r).result }

    val indexed = chunks.flatMap(chunk => {
      if (sentence.contains(chunk)) {
        val index = sentence.indexOf(chunk)
        var tokenIndexBegin = 0
        for (i <- 0 until index) {
          if (sentence(i) == ' ')
            tokenIndexBegin += 1
        }
        val tokenIndexEnd = tokenIndexBegin + chunk.split(" ").length - 1
        Some(IndexedChunk(tokenIndexBegin, tokenIndexEnd))
      } else {
        None
      }
    })

    indexed.map { marked => {
      AssertionAnnotationWithLabel(label, marked.chunkBegin, marked.chunkEnd)
    }}
  }
}