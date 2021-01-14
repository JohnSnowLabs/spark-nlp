package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._
import scala.collection.immutable.ListMap
import scala.collection.mutable

/**
  * Language Identification by using Deep Neural Network
  * Total params: 247,607
  * Trainable params: 247,607
  *
  * @param tensorflow           LanguageDetectorDL Model wrapper with TensorFlow Wrapper
  * @param configProtoBytes     Configuration for TensorFlow session
  *
  **/
class TensorflowLD(val tensorflow: TensorflowWrapper,
                   configProtoBytes: Option[Array[Byte]] = None
                  ) extends Serializable {

  private val inputKey = "inputs:0"
  private val outputKey = "softmax_output_final/Softmax:0"

  def tag(inputs: Array[Array[Float]], inputSize: Int, outputSize: Int): Array[Array[Float]] = {
    val tensors = new TensorResources()

    val tokenBuffers = tensors.createFloatBuffer(inputs.length * inputSize)
    val shape = Array(inputs.length.toLong, inputSize)

    inputs.map { sentence =>
      tokenBuffers.put(sentence)
    }

    tokenBuffers.flip()

    val runner = tensorflow.getSession(configProtoBytes = configProtoBytes).runner

    val tokenTensors = tensors.createFloatBufferTensor(shape, null)

    runner
      .feed(inputKey, tokenTensors)
      .fetch(outputKey)

    val outs = runner.run().asScala
    val predictions = TensorResources.extractFloats(outs.head).grouped(outputSize).toArray

    tensors.clearSession(outs)
    tensors.clearTensors()
    tokenBuffers.clear()

    predictions

  }

  def cleanText(docs: List[String]): List[String] = {
    val rmChars = "@#,.0123456789()-:;\"$%^&*<>+-_=～۰۱۲۳۴۵۶۷۸۹()＄:;\\}\\{｡｢･\\[\\]\\t\\n\\|\\/\\{"
    docs.map(_.replaceAll(rmChars, " "))
  }

  def calculateLanguageIdentification(
                                       documents: Seq[Sentence],
                                       alphabets: Map[String, Int],
                                       languages: Map[String, Int],
                                       threshold: Float = 0.6f,
                                       thresholdLabel: String = "Unknown",
                                       coalesceSentences: Boolean = false
                                     ): Array[Annotation] = {

    val maxSentenceLength = 240
    val orderedAlphabets = ListMap(alphabets.toSeq.sortBy(_._2):_*)
    val orderedLanguages = ListMap(languages.toSeq.sortBy(_._2):_*)

    val sentences = documents.map{ x=>
      val chars = cleanText(x.content.map(_.toString).toList).take(maxSentenceLength)
      val trueCounts = mutable.LinkedHashMap[String, Float]()
      orderedAlphabets.map(x=>trueCounts.put(x._1, 0f))
      chars.foreach{char =>
        if(orderedAlphabets.contains(char)) {
          trueCounts(char) = trueCounts.getOrElse(char, 0f) + 1f
        }
      }
      trueCounts.values.toArray
    }.toArray

    val inputDimension = orderedAlphabets.toArray.length
    val outputDimension = orderedLanguages.toArray.length

    val scores = tag(sentences, inputDimension, outputDimension)
    val langLabels = orderedLanguages.map(x=>x._1.mkString).toArray
    val outputs = scores.map(x=>x.zip(langLabels))

    if (coalesceSentences){

      val avgScores = outputs.flatMap(x=>x.toList).groupBy(_._2).mapValues(_.map(_._1).sum/outputs.length)
      val maxResult = avgScores.maxBy(_._2)
      val finalLabel = if(maxResult._2 >= threshold) maxResult._1 else thresholdLabel

      Array(
        Annotation(
          annotatorType = AnnotatorType.LANGUAGE,
          begin = documents.head.start,
          end = documents.last.end,
          result = finalLabel,
          metadata = Map("sentence" -> documents.head.index.toString) ++ avgScores.flatMap(x => Map(x._1 -> x._2.toString))
        )
      )

    } else {
      outputs.zip(documents).map{ case(score, sentence)=>
        val maxResult = score.maxBy(_._1)
        val finalLabel = if(maxResult._1 >= threshold) maxResult._2 else thresholdLabel

        Annotation(
          annotatorType = AnnotatorType.LANGUAGE,
          begin = sentence.start,
          end = sentence.end,
          result = finalLabel,
          metadata = Map("sentence" -> sentence.index.toString) ++ score.flatMap(x => Map(x._2 -> x._1.toString))
        )
      }
    }
  }
}

