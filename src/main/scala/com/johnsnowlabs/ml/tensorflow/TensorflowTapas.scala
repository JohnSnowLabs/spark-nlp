package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.sign.ModelSignatureConstants
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece, WordpieceTokenizedSentence}
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.parse
import org.tensorflow.ndarray.buffer.IntDataBuffer

import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoField
import scala.util.Try

case class TapasCellDate(day: Option[Int], month: Option[Int], year: Option[Int])

object TapasCellDate {
      def Empty(): TapasCellDate  = {
            TapasCellDate(None, None, None)
      }
}

case class TapasCellValue(number: Option[Float], date: TapasCellDate)

case class TapasNumericValueSpan(begin: Int, end: Int, value: TapasCellValue) {

      def valueId: String = {
            if (value.number.isDefined)
                  value.number.toString
            else if (value.date.day.isDefined || value.date.month.isDefined || value.date.year.isDefined)
                  Array(
                        if (value.date.day.isDefined) value.date.day.get.toString else "NA",
                        if (value.date.month.isDefined) value.date.month.get.toString else "NA",
                        if (value.date.year.isDefined) value.date.year.get.toString else "NA"
                  ).mkString("@")
            else TapasNumericValueSpan.emptyValueId
      }
}

object TapasNumericValueSpan {
      val emptyValueId = "NA"
}

case class Table(header: Array[String], rows: Array[Array[String]])

case class TapasInputData(
                              inputIds: Array[Int],
                              attentionMask: Array[Int],
                              segmentIds: Array[Int],
                              columnIds: Array[Int],
                              rowIds: Array[Int],
                              prevLabels: Array[Int],
                              columnRanks: Array[Int],
                              invertedColumnRanks: Array[Int],
                              numericRelations: Array[Int]
                              )

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

      val DT_FORMATTERS = Array(
            ("MMMM", "\\w+".r),
            ("yyyy", "\\d{4}".r),
            ("yyyy's'", "\\d{4}s".r),
            ("MMM yyyy", "\\w{3}\\s+\\d{4}".r),
            ("MMMM yyyy", "\\w+\\s+\\d{4}".r),
            ("MMMM d", "\\w+\\s+\\d{1,2}".r),
            ("MMM d", "\\w{3}\\s+\\d{1,2}".r),
            ("d MMMM", "\\d{1,2}\\s+\\w+".r),
            ("d MMM", "\\d{1,2}\\s+\\w{3}".r),
            ("MMMM d, yyyy", "\\w+\\s+\\d{1,2},\\s*\\d{4}".r),
            ("d MMMM yyyy", "\\d{1,2}\\s+\\w+\\s+\\d{4}".r),
            ("M-d-yyyy", "\\d{1,2}-\\d{1,2}-\\d{4}".r),
            ("yyyyM-d", "\\d{4}-\\d{1,2}-\\d{1,2}".r),
            ("yyyy MMMM", "\\d{4}\\s+\\w+".r),
            ("d MMM yyyy", "\\d{1,2}\\s+\\w{3}\\s+\\d{4}".r),
            ("yyyy-M-d", "\\d{4}-\\d{1,2}-\\d{1,2}".r),
            ("MMM d, yyyy", "\\w{3}\\s+\\d{1,2},\\s*\\d{4}".r),
            ("d.M.yyyy", "\\d{1,2}\\.\\d{1,2}\\.\\d{4}".r),
            ("E, MMM d", "\\w{3},\\s+\\w{3}\\s+\\d{1,2}".r),
            ("EEEE, MMM d", "\\w+,\\s+\\w{3}\\s+\\d{1,2}".r),
            ("E, MMMM d", "\\w{3},\\s+\\w+\\s+\\d{1,2}".r),
            ("EEEE, MMMM d", "\\w+,\\s+\\w+\\s+\\d{1,2}".r)
      ).map(x => (DateTimeFormatter.ofPattern(x._1), x._2))

      val MIN_YEAR = 1700
      val MAX_YEAR = 2120

      val MIN_NUMBER_OF_ROWS_WITH_VALUES_PROPORTION = 0.5
      val ORDINAL_SUFFIXES = Array("st", "nd", "rd", "th")
      val NUMBER_WORDS = Array("zero",  "one",  "two",  "three",  "four",  "five",
            "six",  "seven", "eight",  "nine",  "ten", "eleven", "twelve")
      val ORDINAL_WORDS = Array("zeroth", "first", "second",  "third",  "fourth",  "fith", "sixth",
            "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth")

      def getAllSpans(text: String, maxNgramLength: Int) = {
            var startIndices: Array[Int] = Array()
            text.zipWithIndex.flatMap{
                  case (ch, i) =>
                        if (ch.isLetterOrDigit) {
                              if (i == 0 || !text(i - 1).isLetterOrDigit)
                                    startIndices = startIndices ++ Array(i)
                              if (((i + 1) == text.length) || !text(i + 1).isLetterOrDigit){
                                    startIndices.drop(startIndices.length-maxNgramLength).map(x => (x, i + 1))
                              } else Array[(Int, Int)]()
                        } else {
                              Array[(Int, Int)]()
                        }
            }.toArray
      }

      def parseNumber(text: String): Option[Float] = {
            var pText = text
            ORDINAL_SUFFIXES
              .foreach(suffix => {
                    if (pText.endsWith(suffix)) {
                          pText = pText.dropRight(suffix.length)
                    }
              })
            pText = pText.replace(",", "")
            Try(pText.toFloat).toOption
      }

      def parseDate(text: String): Option[TapasCellDate] = {
            DT_FORMATTERS
              .filter(dtf => dtf._2.pattern.matcher(text).matches() && Try(dtf._1.parse(text)).isSuccess)
              .map(dtf => {
                    val tempAccessor = dtf._1.parse(text)

                    val day = Try(tempAccessor.get(ChronoField.DAY_OF_MONTH)).toOption
                    val month = Try(tempAccessor.get(ChronoField.MONTH_OF_YEAR)).toOption
                    val year1 = Try(tempAccessor.get(ChronoField.YEAR)).toOption
                    val year2 = if (year1.isDefined) year1 else Try(tempAccessor.get(ChronoField.YEAR_OF_ERA)).toOption
                    val year = if (year2.isDefined && year2.get >= MIN_YEAR && year2.get <= MAX_YEAR) year2 else None

                    if (day.isDefined || month.isDefined || year.isDefined)
                          Some(TapasCellDate(day, month, year))
                    else
                          None
              })
              .filter(_.isDefined)
              .map(x => return x)

            None
      }

      def parseText(text: String): Array[TapasNumericValueSpan] = {

            val spans = collection.mutable.Map[(Int, Int), Array[TapasCellValue]]()

            def addNumberSpan(span: (Int, Int), number: Float) = {
                  if (!spans.contains(span))
                        spans(span) = Array()
                  spans(span) = spans(span) ++ Array(TapasCellValue(Some(number), TapasCellDate.Empty()))
            }
            def addDateSpan(span: (Int, Int), date: TapasCellDate) = {
                  if (!spans.contains(span))
                        spans(span) = Array()
                  spans(span) = spans(span) ++ Array(TapasCellValue(None, date))
            }
            //add numbers
            getAllSpans(text, 1).foreach(span => {
                  val spanText = text.slice(span._1, span._2)
                  val number = parseNumber(spanText)
                  if (number.isDefined)
                        addNumberSpan(span, number.get)
                  NUMBER_WORDS.zipWithIndex.foreach{
                        case (numWord, index) =>
                              if (spanText == numWord)
                                    addNumberSpan(span, index.toFloat)
                  }
                  ORDINAL_WORDS.zipWithIndex.foreach{
                        case (numWord, index) =>
                              if (spanText == numWord)
                                    addNumberSpan(span, index.toFloat)
                  }
            })
            //add dates
            getAllSpans(text, 5).foreach(span => {
                  val spanText = text.slice(span._1, span._2)
                  val date = parseDate(spanText)
                  if (date.isDefined)
                        addDateSpan(span, date.get)
            })

            val sortedSpans = spans.toArray.sortBy(x =>  (x._1._2 - x._1._1, -x._1._1)).reverse
            var selectedSpans = collection.mutable.ArrayBuffer[((Int, Int), Array[TapasCellValue])]()
            sortedSpans.foreach{
                  case (span, values) =>
                        if (!selectedSpans.map(_._1).exists(selectedSpan => selectedSpan._1 <= span._1 && span._2 <= selectedSpan._2)){
                              selectedSpans = selectedSpans ++ Array((span, values))
                        }
            }
            selectedSpans
              .sortBy(x => x._1._1)
              .flatMap{
                    case (span, values) =>
                          values.map(value => TapasNumericValueSpan(span._1, span._2, value))
              }.toArray
      }

      def encodeTapasData(
                           questionAnnotations: Seq[Annotation],
                           tableAnnotation: Annotation,
                           caseSensitive: Boolean,
                           maxSentenceLength: Int): Seq[TapasInputData] = {

            implicit val formats = DefaultFormats

            val basicTokenizer = new BasicTokenizer(caseSensitive = true, hasBeginEnd = false)
            val encoder = new WordpieceEncoder(vocabulary)

            val questionInputIds = questionAnnotations.map(question => {

                  val sentence = new Sentence(
                        start = 0,
                        end = question.result.length,
                        content = question.result,
                        index = 0)
                  val tokens = basicTokenizer.tokenize(sentence)
                  (if (caseSensitive)
                        tokens
                  else
                        tokens.map(x => IndexedToken(x.token.toLowerCase(), x.begin, x.end))
                    ).flatMap(token => encoder.encode(token)).map(_.pieceId)
            })
            val maxQuestionLength = questionInputIds.map(_.length).max

            val table = parse(tableAnnotation.result).extract[Table]

            val inputIds = collection.mutable.ArrayBuffer[Int]()
            val attentionMask = collection.mutable.ArrayBuffer[Int]()
            val segmentIds = collection.mutable.ArrayBuffer[Int]()
            val columnIds = collection.mutable.ArrayBuffer[Int]()
            val rowIds = collection.mutable.ArrayBuffer[Int]()
            val prevLabels = collection.mutable.ArrayBuffer[Int]()
            val columnRanks = collection.mutable.ArrayBuffer[Int]()
            val invertedColumnRanks = collection.mutable.ArrayBuffer[Int]()
            val numericRelations = collection.mutable.ArrayBuffer[Int]()

            table.header.indices.foreach(colIndex => {
                  val sentence = new Sentence(
                        start = 0,
                        end = table.header(colIndex).length,
                        content = table.header(colIndex),
                        index = colIndex)
                  val tokens = basicTokenizer.tokenize(sentence)
                  val columnInputIds = if (caseSensitive)
                        tokens
                  else
                        tokens.map(x => IndexedToken(x.token.toLowerCase(), x.begin, x.end))

                  columnInputIds.flatMap(token => encoder.encode(token)).foreach( x => {
                        inputIds.append(x.pieceId)
                        attentionMask.append(1)
                        segmentIds.append(1)
                        columnIds.append(colIndex + 1)
                        rowIds.append(0)
                        prevLabels.append(0)
                        columnRanks.append(0)
                        invertedColumnRanks.append(0)
                        numericRelations.append(0)
                  })

            })

            val tableCellValues = collection.mutable.Map[Int, Array[(TapasNumericValueSpan, Int, Array[Int])]]()

            table.rows.indices.map(rowIndex => {
                  table.header.indices.map(colIndex => {
                        val cellText = table.rows(rowIndex)(colIndex)
                        val sentence = new Sentence(
                              start = 0,
                              end = cellText.length,
                              content = cellText,
                              index = 0)
                        val tokens = basicTokenizer.tokenize(sentence)
                        val cellInputIds = (if (caseSensitive)
                              tokens
                        else
                              tokens.map(x => IndexedToken(x.token.toLowerCase(), x.begin, x.end))
                        )
                          .flatMap(token => encoder.encode(token))

                        cellInputIds.foreach(x => {
                              inputIds.append(x.pieceId)
                              segmentIds.append(1)
                              columnIds.append(colIndex + 1)
                              rowIds.append(rowIndex + 1)
                              prevLabels.append(0)
                              columnRanks.append(0)
                              invertedColumnRanks.append(0)
                              numericRelations.append(0)//TODO
                        })

                        val tapasNumValuesWithTokenIndices = parseText(cellText).map(numValue =>
                              (numValue,
                                rowIndex,
                                cellInputIds
                                  .zipWithIndex
//                                  .filter(id => id._1.begin>=numValue.begin && id._1.end <= numValue.end)
                                  .map(_._2 + (inputIds.length - cellInputIds.length))
                              ))
                        tableCellValues(colIndex) = tableCellValues.getOrElse(colIndex, Array()) ++ tapasNumValuesWithTokenIndices

                  }).toArray
            }).toArray

            tableCellValues.foreach{
                  case (colIndex, values) =>
                        val rowsWithNumberValues = values.filter(x => x._1.value.number.isDefined).map(_._2).distinct.length
                        val rowsWithDateValues = values.filter(x => x._1.value.number.isEmpty).map(_._2).distinct.length

                        val sortedValues = if (rowsWithNumberValues >= math.max(rowsWithDateValues, MIN_NUMBER_OF_ROWS_WITH_VALUES_PROPORTION * table.rows.length)) {
                              values.sortWith({
                                    (v1, v2) => {
                                          val n1 = v1._1.value.number.getOrElse(Float.MinValue)
                                          val n2 = v2._1.value.number.getOrElse(Float.MinValue)
                                          n1 < n2
                                    }
                              })
                        } else if (rowsWithDateValues >= math.max(rowsWithNumberValues, MIN_NUMBER_OF_ROWS_WITH_VALUES_PROPORTION * table.rows.length)) {
                              values.sortWith({
                                    (v1, v2) => {
                                          val day1 = v1._1.value.date.day.getOrElse(Int.MinValue)
                                          val day2 = v2._1.value.date.day.getOrElse(Int.MinValue)
                                          val month1 = v1._1.value.date.month.getOrElse(Int.MinValue)
                                          val month2 = v2._1.value.date.month.getOrElse(Int.MinValue)
                                          val year1 = v1._1.value.date.year.getOrElse(Int.MinValue)
                                          val year2 = v2._1.value.date.year.getOrElse(Int.MinValue)
                                          (year1 < year2) || (month1 < month2) || (day1 < day2)
                                    }
                              })
                        } else Array[(TapasNumericValueSpan, Int, Array[Int])]()
                        if (!sortedValues.isEmpty) {
                              var rank = 0
                              val curValue = TapasNumericValueSpan.emptyValueId
                              val sortedValuesWithRanks = sortedValues
                                .map(x => {
                                      if (x._1.valueId != curValue) {
                                            rank = rank + 1
                                      }
                                      x._3.foreach(inputIndex => columnRanks(inputIndex) = rank)
                                      (x, rank)
                                })
                              val maxRank = sortedValuesWithRanks.map(_._2).max
                              sortedValuesWithRanks.foreach(x =>
                                    x._1._3.foreach(inputIndex => invertedColumnRanks(inputIndex) = maxRank - x._2 + 1))
                        }
            }

            questionInputIds.map(qIds => {

                  val emptyTokenTypes = Array.fill(qIds.length + 2)(0)
                  val padding = Array.fill(maxQuestionLength - qIds.length)(0)

                  def setMaxSentenceLimit(vector: Array[Int]): Array[Int] = {
                        vector.slice(0, math.min(maxSentenceLength, vector.length))
                  }

                  TapasInputData(
                        inputIds = setMaxSentenceLimit(
                              Array(sentenceStartTokenId) ++ qIds ++ Array(sentenceEndTokenId) ++ inputIds ++ padding),
                        attentionMask = setMaxSentenceLimit(emptyTokenTypes.map(_ => 1) ++ attentionMask ++ padding),
                        segmentIds = setMaxSentenceLimit(emptyTokenTypes ++ segmentIds ++ padding),
                        columnIds = setMaxSentenceLimit(emptyTokenTypes ++ columnIds ++ padding),
                        rowIds = setMaxSentenceLimit(emptyTokenTypes ++ rowIds ++ padding),
                        prevLabels = setMaxSentenceLimit(emptyTokenTypes ++ prevLabels ++ padding),
                        columnRanks = setMaxSentenceLimit(emptyTokenTypes ++ columnRanks ++ padding),
                        invertedColumnRanks = setMaxSentenceLimit(emptyTokenTypes ++ invertedColumnRanks ++ padding),
                        numericRelations = setMaxSentenceLimit(emptyTokenTypes ++ numericRelations ++ padding))
            })

      }

      def tagTapasSpan(batch: Seq[TapasInputData]): (Array[Array[Float]], Array[Array[Float]]) = {

            val tensors = new TensorResources()

            val maxSentenceLength = batch.head.inputIds.length
            val batchLength = batch.length

            val tokenBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
            val maskBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength)
            val segmentBuffers: IntDataBuffer = tensors.createIntBuffer(batchLength * maxSentenceLength * 7)

            batch
              .zipWithIndex
              .foreach {
                    case (input, idx) =>
                          val offset = idx * maxSentenceLength
                          tokenBuffers.offset(offset).write(input.inputIds)
                          maskBuffers.offset(offset).write(input.attentionMask)
                          segmentBuffers.offset(offset).write(
                                input.segmentIds ++ input.rowIds++ input.columnIds  ++ input.prevLabels
                                  ++ input.columnRanks ++ input.invertedColumnRanks ++ input.numericRelations)
              }

            val session = tensorflowWrapper.getTFSessionWithSignature(
                  configProtoBytes = configProtoBytes,
                  savedSignatures = signatures,
                  initAllTables = false)
            val runner = session.runner

            val tokenTensors = tensors.createIntBufferTensor(Array(batchLength.toLong, maxSentenceLength), tokenBuffers)
            val maskTensors = tensors.createIntBufferTensor(Array(batchLength.toLong, maxSentenceLength), maskBuffers)
            val segmentTensors = tensors.createIntBufferTensor(Array(batchLength.toLong, maxSentenceLength, 7), segmentBuffers)

            runner
              .feed(
                    _tfBertSignatures.getOrElse(ModelSignatureConstants.InputIds.key, "missing_input_id_key"),
                    tokenTensors)
              .feed(
                    _tfBertSignatures.getOrElse(
                          ModelSignatureConstants.AttentionMask.key,
                          "missing_input_mask_key"),
                    maskTensors)
              .feed(
                    _tfBertSignatures.getOrElse(
                          ModelSignatureConstants.TokenTypeIds.key,
                          "missing_segment_ids_key"),
                    segmentTensors)
              .fetch(_tfBertSignatures
                .getOrElse(ModelSignatureConstants.EndLogitsOutput.key, "missing_end_logits_key"))
              .fetch(_tfBertSignatures
                .getOrElse(ModelSignatureConstants.StartLogitsOutput.key, "missing_start_logits_key"))

            val outs = runner.run()

            batch.foreach(tapasData => {
                  println("Input Ids:")
                  println(tapasData.inputIds.slice(0, 50).map(_.toString).mkString(" "))
                  println("Attention Mask:")
                  println(tapasData.attentionMask.slice(0, 50).map(_.toString).mkString(" "))
                  println("Segment Ids:")
                  println(tapasData.segmentIds.slice(0, 50).map(_.toString).mkString(" "))
                  println("Column Ids:")
                  println(tapasData.columnIds.slice(0, 50).map(_.toString).mkString(" "))
                  println("RowIds:")
                  println(tapasData.rowIds.slice(0, 50).map(_.toString).mkString(" "))
                  println("prevLabels:")
                  println(tapasData.prevLabels.slice(0, 50).map(_.toString).mkString(" "))
                  println("columnRanks:")
                  println(tapasData.columnRanks.slice(0, 50).map(_.toString).mkString(" "))
                  println("invertedColumnRanks:")
                  println(tapasData.invertedColumnRanks.slice(0, 50).map(_.toString).mkString(" "))
                  println("numericRelations:")
                  println(tapasData.numericRelations.slice(0, 50).map(_.toString).mkString(" "))
            })
            (Array(), Array())
      }

      def predictTapasSpan(
                            questions: Seq[Annotation],
                            table: Annotation,
                            maxSentenceLength: Int,
                            caseSensitive: Boolean): Seq[Annotation] = {

            val tapasData = encodeTapasData(
                  questionAnnotations = questions, tableAnnotation = table, caseSensitive = caseSensitive, maxSentenceLength = maxSentenceLength)

            tagTapasSpan(batch = tapasData)

            Seq()
      }

}
