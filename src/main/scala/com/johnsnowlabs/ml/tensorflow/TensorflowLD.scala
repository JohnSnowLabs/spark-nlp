package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  *
  *
  *
  *
  *
  * @param tensorflow           LanguageDetectorDL Model wrapper with TensorFlow Wrapper
  * @param configProtoBytes     Configuration for TensorFlow session
  *
  *                             Paper:  [[ https://arxiv.org/abs/1810.04805]]
  *
  *                             Source:  [[https://github.com/google-research/bert]]
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

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes).runner

    val tokenTensors = tensors.createFloatBufferTensor(shape, tokenBuffers)

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
                                       alphabets: Map[String, Int],
                                       languages: Map[String, Int]
                                     ): Array[Annotation] = {

    val maxSentenceLength = 512

    val sentences = documents.map{ x=>
      val chars = x.content.map(_.toString).toList.take(maxSentenceLength)
      val trueCounts = mutable.LinkedHashMap[String, Int]()
      alphabets.map(x=>trueCounts.put(x._1, 0))
      chars.foreach{char =>
        if(alphabets.contains(char)) {
          trueCounts(char) = trueCounts.getOrElse(char, 0) + 1
        }
      }
      trueCounts.map(x=>x._2.toFloat).toArray
    }.toArray

    val inputDimension = alphabets.toArray.length
    val outputDimension = languages.toArray.length

    val scores = tag(sentences, inputDimension, outputDimension)
    val langLabels = languages.map(x=>x._1.mkString).toArray
    val outputs = scores.map(x=>x.zip(langLabels))

    outputs.zip(documents).map{case(score, sentence)=>
      val maxResult = score.maxBy(_._1)

      Annotation(
        annotatorType = AnnotatorType.LANGUAGE,
        begin = sentence.start,
        end = sentence.end,
        result = maxResult._2,
        metadata = Map("sentence" -> sentence.index.toString) ++ score.flatMap(x => Map(x._2 -> x._1.toString))
      )
    }
  }
}

