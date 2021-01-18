package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._
import scala.collection.immutable.ListMap
import scala.collection.mutable

/**
  * Language Identification and Detection by using CNNs and RNNs architectures in TensowrFlow
  *
  * The models are trained on large datasets such as Wikipedia and Tatoeba
  * The output is a language code in Wiki Code style: https://en.wikipedia.org/wiki/List_of_Wikipedias
  *
  *
  * @param tensorflow           LanguageDetectorDL Model wrapper with TensorFlow Wrapper
  * @param configProtoBytes     Configuration for TensorFlow session
  * @param orderedLanguages     ordered ListMap of language codes detectable by this trained model
  * @param orderedAlphabets     ordered ListMap of alphabets to be used to encode the inputs
  *
  **/
class TensorflowLD(val tensorflow: TensorflowWrapper,
                   configProtoBytes: Option[Array[Byte]] = None,
                   orderedLanguages: ListMap[String, Int],
                   orderedAlphabets: ListMap[String, Int]
                  ) extends Serializable {

  private val inputKey = "inputs:0"
  private val outputKey = "output/Softmax:0"
  // LD models from 2.7.0 must be 150 sequences
  private val maxSentenceLength = 150

  def cleanText(docs: List[String]): List[String] = {
    val rmChars = "!\"#$%&()*+,-./:;<=>?@[\\\\]^_`\\{|\\}~\\t\\n"
    docs.map(_.replaceAll(rmChars, "").toLowerCase())
  }

  def encode(docs: Seq[Sentence]): Array[Array[Float]] = {
    val charsArr = orderedAlphabets.keys.toArray

    docs.map{ x =>
      val chars = cleanText(x.content.map(_.toString).toList).take(maxSentenceLength)
      val tokens = mutable.ArrayBuffer[Float]()

      chars.foreach{char =>
        val charID = charsArr.indexOf(char).toFloat
        if(charID >= 0){
          tokens.append(charID + 1.0f)
        }
      }
      val diff = maxSentenceLength - tokens.length
      tokens.toArray ++ Array.fill(diff)(0.0f)
    }.toArray
  }

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

//    val tokenTensors = tensors.createFloatBufferTensor(shape, tokenBuffers)

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

  def calculateLanguageIdentification(
                                       documents: Seq[Sentence],
                                       threshold: Float = 0.01f,
                                       thresholdLabel: String = "unk",
                                       coalesceSentences: Boolean = false
                                     ): Array[Annotation] = {


    val sentences = encode(documents)

    val outputDimension = orderedLanguages.toArray.length

    val scores = tag(sentences, maxSentenceLength, outputDimension)
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

