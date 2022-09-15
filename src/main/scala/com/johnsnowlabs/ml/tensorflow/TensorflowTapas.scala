package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

class TensorflowTapas(
    override val tensorflowWrapper: TensorflowWrapper,
    override val sentenceStartTokenId: Int,
    override val sentenceEndTokenId: Int,
    configProtoBytes: Option[Array[Byte]] = None,
    tags: Map[String, Int],
    signatures: Option[Map[String, String]] = None,
    vocabulary: Map[String, Int])
    extends TensorflowBertClassification(
      tensorflowWrapper = tensorflowWrapper,
      sentenceStartTokenId = sentenceStartTokenId,
      sentenceEndTokenId = sentenceEndTokenId,
      configProtoBytes = configProtoBytes,
      tags = tags,
      signatures = signatures,
      vocabulary = vocabulary) {


      override def predictSpan(
                       documents: Seq[Annotation],
                       maxSentenceLength: Int,
                       caseSensitive: Boolean,
                       mergeTokenStrategy: String = MergeTokenStrategy.vocab): Seq[Annotation] = {

            val questionAnnot = Seq(documents.head)
            val contextAnnot = documents.drop(1)

            val wordPieceTokenizedQuestion =
                  tokenizeDocument(questionAnnot, maxSentenceLength, caseSensitive)
            val wordPieceTokenizedContext =
                  tokenizeDocument(contextAnnot, maxSentenceLength, caseSensitive)

            val encodedInput =
                  encodeSequence(wordPieceTokenizedQuestion, wordPieceTokenizedContext, maxSentenceLength)
            val (startLogits, endLogits) = tagSpan(encodedInput)

            val startScores = startLogits.transpose.map(_.sum / startLogits.length)
            val endScores = endLogits.transpose.map(_.sum / endLogits.length)

            val startIndex = startScores.zipWithIndex.maxBy(_._1)
            val endIndex = endScores.zipWithIndex.maxBy(_._1)

            val allTokenPieces =
                  wordPieceTokenizedQuestion.head.tokens ++ wordPieceTokenizedContext.flatMap(x => x.tokens)
            val decodedAnswer = allTokenPieces.slice(startIndex._2 - 2, endIndex._2 - 1)
            val content =
                  mergeTokenStrategy match {
                        case MergeTokenStrategy.vocab =>
                              decodedAnswer.filter(_.isWordStart).map(x => x.token).mkString(" ")
                        case MergeTokenStrategy.sentencePiece =>
                              val token = ""
                              decodedAnswer
                                .map(x =>
                                      if (x.isWordStart) " " + token + x.token
                                      else token + x.token)
                                .mkString("")
                                .trim
                  }

            Seq(
                  Annotation(
                        annotatorType = AnnotatorType.CHUNK,
                        begin = 0,
                        end = if (content.isEmpty) 0 else content.length - 1,
                        result = content,
                        metadata = Map(
                              "sentence" -> "0",
                              "chunk" -> "0",
                              "start" -> startIndex._2.toString,
                              "start_score" -> startIndex._1.toString,
                              "end" -> endIndex._2.toString,
                              "end_score" -> endIndex._1.toString,
                              "score" -> ((startIndex._1 + endIndex._1) / 2).toString)))

      }

}
