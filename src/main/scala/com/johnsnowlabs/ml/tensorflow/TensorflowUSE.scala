package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common._
import scala.collection.JavaConverters._


class TensorflowUSE(val tensorflow: TensorflowWrapper,
                    configProtoBytes: Option[Array[Byte]] = None
                   ) extends Serializable {

  private val inputKey = "input"
  private val outPutKey = "output"

  def calculateEmbeddings(sentences: Seq[Sentence]): Seq[Annotation] = {

    val tensors = new TensorResources()
    val batchSize = sentences.length

    val sentencesBytes = sentences.map{ x=>
      x.content.getBytes("UTF-8")
    }.toArray

    val sentenceTensors = tensors.createTensor(sentencesBytes)

    val runner = tensorflow.getTFHubSession(configProtoBytes = configProtoBytes).runner

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
