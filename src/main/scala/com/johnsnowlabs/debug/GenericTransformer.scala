package com.johnsnowlabs.debug

import com.johnsnowlabs.nlp.annotator.RegexTokenizer
import com.johnsnowlabs.nlp.annotators.common.{TokenizedSentence, TokenizedWithSentence}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

object GenericTransformer {

  val modelPath = "./bert_for_multiple_choice.onnx"
  val vocabularyPath = "./bert_base_uncased_multiple_choice/vocab.txt"
  val sentenceStartToken = "[CLS]"
  val sentencePadToken = "[PAD]"
  val sentenceEndToken = "[SEP]"
  val maxSentenceLength = 512
  val caseSensitive = false

  def tokenizedWithSentence(text: String): Array[TokenizedSentence] = {
    val annotation = Seq(Annotation(AnnotatorType.DOCUMENT, 0, text.length, text, Map("sentence" -> "0")))
    val regexTokenizer = new RegexTokenizer().setInputCols("document").setOutputCol("token")
    val result = annotation ++ regexTokenizer.annotate(annotation)
    TokenizedWithSentence.unpack(result).toArray
  }

  def predict(text: String): Unit = {
    val tokenizedSentences = tokenizedWithSentence(text)
    val bertTokenizer = getTransformerTokenizer()

    val onnxModel = new ImportONNXModel(modelPath, bertTokenizer)
    val logits = onnxModel.computeLogits(tokenizedSentences, 1, 150, false)
    println()
  }


  //For Question Answering or similar tasks
  def predictWithContext(text: String, context: String): Seq[Annotation] = {
    val textAnnotation = Seq(Annotation(AnnotatorType.DOCUMENT, 0, text.length, text, Map("sentence" -> "0")))
    val contextAnnotation = Seq(Annotation(AnnotatorType.DOCUMENT, 0, context.length, context, Map("sentence" -> "0")))
    val annotations = contextAnnotation ++ textAnnotation
    predictSpan(annotations)
  }

  def predictSpan(documents: Seq[Annotation]): Seq[Annotation] = {
    val questionAnnot = Seq(documents.head)
    val contextAnnot = documents.drop(1)

    val tokenizer = getTransformerTokenizer()

    val wordPieceTokenizedQuestion = tokenizer.tokenizeDocument(questionAnnot, maxSentenceLength, caseSensitive)
    val wordPieceTokenizedContext = tokenizer.tokenizeDocument(contextAnnot, maxSentenceLength, caseSensitive)

    val encodedInput =
      tokenizer.encodeSequence(wordPieceTokenizedQuestion, wordPieceTokenizedContext, maxSentenceLength)

    val onnxModel = new ImportONNXModel(modelPath, tokenizer)
    val logits = onnxModel.computeLogitsWithContext(encodedInput, maxSentenceLength)


    Seq()
  }

  def predictWithContextAndMultipleChoice(context: String, choices: Array[String]): Seq[Annotation] = {
    val contextAnnotation = Seq(Annotation(AnnotatorType.DOCUMENT, 0, context.length, context, Map("sentence" -> "0")))
    predictSpanMultipleChoice(contextAnnotation, choices)
  }

  private def predictSpanMultipleChoice(question: Seq[Annotation], choices: Seq[String]): Seq[Annotation] = {
    val tokenizer = getTransformerTokenizer()
    val wordPieceTokenizedQuestion = tokenizer.tokenizeDocument(question, maxSentenceLength, caseSensitive)

    val inputIds = choices.flatMap{ choice =>
      val choiceAnnotation = Seq(Annotation(AnnotatorType.DOCUMENT, 0, choice.length, choice, Map("sentence" -> "0")))
      val wordPieceTokenizedChoice = tokenizer.tokenizeDocument(choiceAnnotation, maxSentenceLength, caseSensitive)
      val encodedInput = tokenizer.encodeSequence(wordPieceTokenizedQuestion, wordPieceTokenizedChoice, maxSentenceLength)
      encodedInput
    }

    val onnxModel = new ImportONNXModel(modelPath, tokenizer)
    val logits = onnxModel.computeLogitsWithContext(inputIds, maxSentenceLength)

    Seq()
  }

  private def getTransformerTokenizer() = {
    val vocabularyResource =
      new ExternalResource(vocabularyPath, ReadAs.TEXT, Map("format" -> "text"))
    val vocabulary = ResourceHelper.parseLines(vocabularyResource).zipWithIndex.toMap
    val bertTokenizer = new TransformersTokenizer(
      sentenceStartToken,
      sentencePadToken,
      sentenceEndToken,
      vocabulary)

    bertTokenizer
  }


}
