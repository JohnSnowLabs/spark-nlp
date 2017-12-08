package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.embeddings.{AnnotatorWithWordEmbeddings, WordEmbeddings}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema

import scala.collection.mutable
import scala.util.Try

/**
  * Created by jose on 24/11/17.
  */
trait Windowing extends Serializable{

  val before : Int
  val after : Int

  lazy val wordVectors: Option[WordEmbeddings] = None

  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, s: Int, e: Int) : Array[String] = {

    val target = doc.slice(s, e)
    val targetPart = tokenize(target.trim)
    //println(target)

    val leftPart = if (s == 0) Array[String]()
    else tokenize(doc.slice(0, s).trim) //TODO add proper tokenizer here

    val rightPart = if (e == doc.length) Array[String]()
    else tokenize(doc.slice(e, doc.length).trim)

    val (start, leftPadding) =
      if(leftPart.size >= before)
        (leftPart.size - before, Array[String]())
      else
        (0, Array.fill(before - leftPart.length)("empty_marker"))

    val (end, rightPadding) =
      if(targetPart.length - 1 + rightPart.length <= after)
        (rightPart.length, Array.fill(after - (targetPart.length - 1 + rightPart.length))("empty_marker"))
      else
        (after - targetPart.length, Array[String]())

    val leftContext = leftPart.slice(start, leftPart.length)
    val rightContext = rightPart.slice(0, end + 1)

    leftPadding ++ leftContext ++ targetPart ++ rightContext ++ rightPadding
  }

  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, target: String) : Array[String] = {
    val start = doc.indexOf(target)
    val end = start + target.length
    applyWindow(doc, start, end)
  }

  /* same as above, but convert the resulting text in a vector */
  def applyWindowUdf(wvectors: WordEmbeddings, codes: Map[String, Array[Double]]) =
    udf {(doc:String, pos:mutable.WrappedArray[GenericRowWithSchema], start:Int, end:Int, target:String)  =>
      val tmp = applyWindow(doc.toLowerCase, target.toLowerCase).
        flatMap(wvectors.getEmbeddings).map(_.toDouble)

      val empty = Array.fill(3)(0.0)
      val previous = if (start < 3) empty ++ empty ++ empty
      else pos.toArray.slice(start - 3, start).map(_.getString(3)).toArray.flatMap(tag => codes.get(tag).getOrElse(empty))

      var result = if (previous.length == 9) tmp ++ previous else tmp ++ empty ++ empty ++ empty

      val following = if (end + 3 > pos.size) empty ++ empty ++ empty
      else pos.toArray.slice(end, end + 3).map(_.getString(3)).toArray.flatMap(tag => codes.get(tag).getOrElse(empty))

      result = if (following.length == 9) result ++ following else result ++ empty ++ empty ++ empty
      //if(result.length != 4009)
        //println(tmp.length, previous.length)
      Vectors.dense(result)


    }

  /* same as above, but convert the resulting text in a vector */
  def applyWindowUdf(wvectors: WordEmbeddings) =
    udf {(doc:String, target:String)  =>
      val tmp = applyWindow(doc.toLowerCase, target.toLowerCase).
        flatMap(wvectors.getEmbeddings).map(_.toDouble)
      Vectors.dense(tmp)
    }

  /* appends POS tags at the end of the vector */
  def appendPos(codes: Map[String, Array[Double]]) =
    udf {(vector:Vector, pos:mutable.WrappedArray[GenericRowWithSchema], start:Int, end:Int, target:String)  =>
        val empty = Array.fill(9)(0.0)
        val previous = if (start < 3) empty
        else pos.toArray.slice(start - 3, start).map(_.getString(3)).toArray.flatMap(tag => codes.get(tag).getOrElse(empty))

      Vectors.dense(vector.toArray ++ empty)
    }

  val punctuation = Seq(".", ":", ";", ",", "?", "!", "+", "-", "_", "(", ")", "{",
    "}", "#", "/", "\\", "\"", "\'", "[", "]", "%", "<", ">", "&", "=")

  /* Tokenize a sentence taking care of punctuation */
  def tokenize(sentence: String) : Array[String] = sentence.split(" ")

}
