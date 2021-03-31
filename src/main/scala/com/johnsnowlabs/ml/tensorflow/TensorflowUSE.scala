package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import scala.collection.JavaConverters._

/**
  * The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/UniversalSentenceEncoderTestSpec.scala]] for further reference on how to use this API.
  *
  * @param tensorflow       USE Model wrapper with TensorFlow Wrapper
  * @param configProtoBytes Configuration for TensorFlow session
  *
  *                         Sources :
  *
  *                         [[https://arxiv.org/abs/1803.11175]]
  *
  *                         [[https://tfhub.dev/google/universal-sentence-encoder/2]]
  */
class TensorflowUSE(val tensorflow: TensorflowWrapper,
                    configProtoBytes: Option[Array[Byte]] = None,
                    loadSP: Boolean = false,
                   ) extends Serializable {

  private val inputKey = "input"
  private val outPutKey = "output"

  def calculateEmbeddings(sentences: Seq[Sentence]): Seq[Annotation] = {

    val tensors = new TensorResources()
    val batchSize = sentences.length

    val sentencesContent = sentences.map{ x=>
      x.content
    }.toArray

    val sentenceTensors = tensors.createTensor(sentencesContent)

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes, loadSP = loadSP).runner

    runner
      .feed(inputKey, sentenceTensors)
      .fetch(outPutKey)

    val outs = runner.run().asScala
    val allEmbeddings = TensorResources.extractFloats(outs.head)

    tensors.clearSession(outs)
    tensors.clearTensors()
    sentenceTensors.close()

    val dim = allEmbeddings.length / batchSize
    val embeddings = allEmbeddings.grouped(dim).toArray

    sentences.zip(embeddings).map { case (sentence, vectors) =>
      Annotation(
        annotatorType = AnnotatorType.SENTENCE_EMBEDDINGS,
        begin = sentence.start,
        end = sentence.end,
        result = sentence.content,
        metadata = Map("sentence" -> sentence.index.toString,
          "token" -> sentence.content,
          "pieceId" -> "-1",
          "isWordStart" -> "true"
        ),
        embeddings = vectors
      )
    }
  }

}
