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
    /*
    val indexed = chunks.flatMap(chunk => {
        if (sentence.contains(chunk)) {
          val tokenIndexBegin = sentence.indexOf(chunk)
          val tokenIndexEnd = tokenIndexBegin + chunk.length - 1
          Some(IndexedChunk(tokenIndexBegin, tokenIndexEnd))
        } else {
          None
        }
      })

    val indexed = documents.map(Annotation(_).result).zipAll(chunks.map(Annotation(_).result), "", "")
      .map { case (doc, chunk) =>
        if (chunk.isEmpty) {
          IndexedChunk("", 0, 0)
        } else if (doc.isEmpty) {
          /** More than one chunk per document*/
          lastIC.get
        } else {
          require(doc.contains(chunk), s"Chunk: $chunk is not a substring of document: $doc")
          val index = doc.indexOf(chunk)
          var tokenIndexBegin = 0
          for (i <- 0 until index) {
            if (doc(i) == ' ')
              tokenIndexBegin += 1
          }
          val tokenIndexEnd = tokenIndexBegin + chunk.split(" ").length - 1
          val ic = IndexedChunk(doc, tokenIndexBegin, tokenIndexEnd)
          lastIC = Some(ic)
          ic
        }
      }
*/
    indexed.map { marked => {
      AssertionAnnotationWithLabel(label, marked.chunkBegin, marked.chunkEnd)
    }}
  }
}