package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import scala.util.Random

import scala.collection.mutable

/**
  * Created by jose on 24/11/17.
  */
trait Windowing extends Serializable {

  /* */
  val before : Int
  val after : Int

  lazy val wordVectors: Option[WordEmbeddings] = None
  val random = new Random()

  /* TODO: create a tokenizer class */
  /* these match the behavior we had when tokenizing sentences for word embeddings */
  val punctuation = Seq(".", ":", ";", ",", "?", "!", "+", "-", "_", "(", ")", "{",
    "}", "#", "mg/kg", "ml", "m2", "cm", "/", "\\", "\"", "'", "[", "]", "%", "<", ">", "&", "=")

  val percent_regex = """([0-9]{1,2}\.[0-9]{1,2}%|[0-9]{1,3}%)"""
  val number_regex = """([0-9]{1,6})"""


  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, s: Int, e: Int): (Array[String], Array[String], Array[String])  = {

    val target = doc.slice(s, e)
    val targetPart = tokenize(target.trim)
    val leftPart = if (s == 0) Array[String]()
    else tokenize(doc.slice(0, s).trim)

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

    (leftPadding ++ leftContext, targetPart, rightContext ++ rightPadding)
  }

  /* apply window, pad/truncate sentence according to window */
  def applyWindow(doc: String, target: String) : (Array[String], Array[String], Array[String])= {
    val start = doc.indexOf(target)
    val end = start + target.length
    applyWindow(doc, start, end)
  }


  def applyWindow(wvectors: WordEmbeddings) (doc:String, targetTerm:String, s:Int, e:Int) : Array[Double]  = {
    val tokens = doc.split(" ").filter(_!="")

    /* now start and end are indexes in the doc string */
    val start = tokens.slice(0, s).map(_.length).sum +
      tokens.slice(0, s).size // account for spaces
    val end = start + tokens.slice(s, e + 1).map(_.length).sum +
        tokens.slice(s, e + 1).size  // account for spaces

    val (l, t, r) = applyWindow(doc.toLowerCase, start, end)

    l.flatMap(w => normalize(wvectors.getEmbeddings(w).map(_.toDouble))) ++
      t.flatMap(w =>  normalize(wvectors.getEmbeddings(w).map(_.toDouble))) ++
      r.flatMap(w =>  normalize(wvectors.getEmbeddings(w).map(_.toDouble)))
  }

  def applyWindowUdf(wvectors: WordEmbeddings, codes: Map[String, Array[Double]]) =
    udf {(doc:String, pos:mutable.WrappedArray[GenericRowWithSchema], start:Int, end:Int, targetTerm:String)  =>
      val (l, t, r) = applyWindow(doc.toLowerCase, targetTerm.toLowerCase)
      var target = Array(0.1, -0.1)
      var nonTarget = Array(-0.1, 0.1)
      l.flatMap(w => wvectors.getEmbeddings(w)).map(_.toDouble) ++
        t.flatMap(w => wvectors.getEmbeddings(w).map(_.toDouble) ).map(_.toDouble) ++
        r.flatMap(w => wvectors.getEmbeddings(w).map(_.toDouble)  ).map(_.toDouble)
    }

  var tt = Array(random.nextGaussian(), random.nextGaussian())
  var nt = Array(-random.nextGaussian(), -random.nextGaussian())

  def applyWindowUdf(wvectors: WordEmbeddings) =
    //here s and e are token number for start and end of target when split on " "
    udf { (doc:String, targetTerm:String, s:Int, e:Int) =>
      Vectors.dense(applyWindow(wvectors)(doc, targetTerm, s, e))
    }


  def l2norm(xs: Array[Double]):Double = {
    import math._
    sqrt(xs.map{ x => pow(x, 2)}.sum)
  }

  def normalize(vec: Array[Double]) : Array[Double] = {
    val norm = l2norm(vec) + 1.0
    vec.map(element => element / norm)
  }

  /* Tokenize a sentence taking care of punctuation */
  def tokenize(sent: String) : Array[String] = {
    var tmp = sent

    // replace special characters
    punctuation.foreach(c => tmp = tmp.replace(c, " " + c + " "))
    tmp = tmp.replace(",", " ")
    tmp = tmp.replace("&quot;", "\"")
    tmp = tmp.replace("&apos;", " ' ")
    tmp.split(" ").map(_.trim).filter(_ != "")
  }
}
