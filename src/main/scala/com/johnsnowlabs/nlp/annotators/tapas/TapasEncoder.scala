package com.johnsnowlabs.nlp.annotators.tapas

import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TableData}
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}

import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoField
import scala.util.matching.Regex

import scala.util.Try

case class TapasCellDate(day: Option[Int], month: Option[Int], year: Option[Int]) {
  def isDefined: Boolean = {
    day.isDefined || month.isDefined || year.isDefined
  }

  def compareTo(other: TapasCellDate): Boolean = TapasCellDate.compare(this, other)

  def isEqualTo(other: TapasCellDate): Boolean = TapasCellDate.areEqual(this, other)
}

object TapasCellDate {
  def Empty(): TapasCellDate = {
    TapasCellDate(None, None, None)
  }

  def computeNumericRelation(v1: TapasCellDate, v2: TapasCellDate): Int = {
    if (v1.isDefined && v2.isDefined) {
      if (areEqual(v1, v2))
        TapasNumericRelation.EQ
      else if (compare(v1, v2)) TapasNumericRelation.LT
      else TapasNumericRelation.GT

    } else TapasNumericRelation.NONE
  }

  def areEqual(v1: TapasCellDate, v2: TapasCellDate): Boolean = {
    val dayMatch = v1.day.getOrElse(Int.MinValue) == v2.day.getOrElse(Int.MinValue)
    val monthMatch = v1.month.getOrElse(Int.MinValue) == v2.month.getOrElse(Int.MinValue)
    val yearMatch = v1.year.getOrElse(Int.MinValue) == v2.year.getOrElse(Int.MinValue)
    dayMatch && monthMatch && yearMatch
  }

  def compare(v1: TapasCellDate, v2: TapasCellDate): Boolean = {
    val dayDiff = Try(v1.day.get - v2.day.get).getOrElse(0)
    val monthDiff = Try(v1.month.get - v2.month.get).getOrElse(0)
    val yearDiff = Try(v1.year.get - v2.year.get).getOrElse(0)

    (dayDiff < 0) || (monthDiff < 0) || (yearDiff < 0)
  }
}

case class TapasCellValue(number: Option[Float], date: TapasCellDate) {
  def compareTo(other: TapasCellValue): Boolean = TapasCellValue.compare(this, other)
  def getNumericRelationTo(other: TapasCellValue): Int =
    TapasCellValue.computeNumericRelation(this, other)
}

object TapasCellValue {

  def compareNumbers(v1: TapasCellValue, v2: TapasCellValue): Boolean = {
    val n1 = v1.number.getOrElse(Float.MinValue)
    val n2 = v2.number.getOrElse(Float.MinValue)
    n1 < n2
  }

  def compareDates(v1: TapasCellValue, v2: TapasCellValue): Boolean = v1.date.compareTo(v2.date)

  def compare(v1: TapasCellValue, v2: TapasCellValue): Boolean = {
    if (v1.number.isDefined && v2.number.isDefined) {
      compareNumbers(v1, v2)
    } else if (v1.date.isDefined && v1.date.isDefined) {
      compareDates(v1, v2)
    } else {
      false
    }
  }

  def computeNumericRelation(v1: TapasCellValue, v2: TapasCellValue): Int = {
    if (v1.number.isDefined && v2.number.isDefined) (v1.number.get - v2.number.get) match {
      case 0 => return TapasNumericRelation.EQ
      case x if x > 0 => return TapasNumericRelation.GT
      case x if x < 0 => return TapasNumericRelation.LT
    }
    else if (v1.date.isDefined && v2.date.isDefined) {
      return TapasCellDate.computeNumericRelation(v1.date, v2.date)
    }

    TapasNumericRelation.NONE
  }
}

case class TapasNumericValueSpan(begin: Int, end: Int, value: TapasCellValue) {

  def valueId: String = {
    if (value.number.isDefined)
      value.number.toString
    else if (value.date.day.isDefined || value.date.month.isDefined || value.date.year.isDefined)
      Array(
        if (value.date.day.isDefined) value.date.day.get.toString else "NA",
        if (value.date.month.isDefined) value.date.month.get.toString else "NA",
        if (value.date.year.isDefined) value.date.year.get.toString else "NA").mkString("@")
    else TapasNumericValueSpan.emptyValueId
  }

  def compareTo(other: TapasNumericValueSpan): Boolean =
    TapasCellValue.compare(this.value, other.value)
}

object TapasNumericValueSpan {
  val emptyValueId = "NA"
}

object TapasNumericRelation {
  val NONE = 0
  val HEADER_TO_CELL = 1 // Connects header to cell.
  val CELL_TO_HEADER = 2 // Connects cell to header.
  val QUERY_TO_HEADER = 3 // Connects query to headers.
  val QUERY_TO_CELL = 4 // Connects query to cells.
  val ROW_TO_CELL = 5 // Connects row to cells.
  val CELL_TO_ROW = 6 // Connects cells to row.
  val EQ = 7 // Annotation value is same as cell value
  val LT = 8 // Annotation value is less than cell value
  val GT = 9 // Annotation value is greater than cell value
}

case class TapasInputData(
    inputIds: Array[Int],
    attentionMask: Array[Int],
    segmentIds: Array[Int],
    columnIds: Array[Int],
    rowIds: Array[Int],
    prevLabels: Array[Int],
    columnRanks: Array[Int],
    invertedColumnRanks: Array[Int],
    numericRelations: Array[Int])

class TapasEncoder(
    val sentenceStartTokenId: Int,
    val sentenceEndTokenId: Int,
    encoder: WordpieceEncoder) {

  protected val NUMBER_PATTERN: Regex =
    "((^|\\s)[+-])?((\\.\\d+)|(\\d+(,\\d\\d\\d)*(\\.\\d*)?))".r

  protected val DT_FORMATTERS: Array[(DateTimeFormatter, Regex)] = Array(
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
    ("EEEE, MMMM d", "\\w+,\\s+\\w+\\s+\\d{1,2}".r)).map(x =>
    (DateTimeFormatter.ofPattern(x._1), x._2))

  protected val MIN_YEAR = 1700
  protected val MAX_YEAR = 2120

  protected val MIN_NUMBER_OF_ROWS_WITH_VALUES_PROPORTION = 0.5f

  protected val ORDINAL_SUFFIXES: Array[String] = Array("st", "nd", "rd", "th")
  protected val NUMBER_WORDS: Array[String] = Array(
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve")
  protected val ORDINAL_WORDS: Array[String] = Array(
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fith",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth")

  protected val AGGREGATIONS = Map(0 -> "NONE", 1 -> "SUM", 2 -> "AVERAGE", 3 -> "COUNT")

  def getAggregationString(aggregationId: Int): String = {
    AGGREGATIONS(aggregationId)
  }

  protected def getAllSpans(text: String, maxNgramLength: Int): Array[(Int, Int)] = {
    var startIndices: Array[Int] = Array()
    text.zipWithIndex.flatMap { case (ch, i) =>
      if (ch.isLetterOrDigit) {
        if (i == 0 || !text(i - 1).isLetterOrDigit)
          startIndices = startIndices ++ Array(i)
        if (((i + 1) == text.length) || !text(i + 1).isLetterOrDigit) {
          startIndices.drop(startIndices.length - maxNgramLength).map(x => (x, i + 1))
        } else Array[(Int, Int)]()
      } else {
        Array[(Int, Int)]()
      }
    }.toArray
  }

  protected def parseNumber(text: String): Option[Float] = {
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

  protected def parseDate(text: String): Option[TapasCellDate] = {
    DT_FORMATTERS
      .filter(dtf => dtf._2.pattern.matcher(text).matches() && Try(dtf._1.parse(text)).isSuccess)
      .map(dtf => {
        val tempAccessor = dtf._1.parse(text)

        val day = Try(tempAccessor.get(ChronoField.DAY_OF_MONTH)).toOption
        val month = Try(tempAccessor.get(ChronoField.MONTH_OF_YEAR)).toOption
        val year1 = Try(tempAccessor.get(ChronoField.YEAR)).toOption
        val year2 =
          if (year1.isDefined) year1 else Try(tempAccessor.get(ChronoField.YEAR_OF_ERA)).toOption
        val year =
          if (year2.isDefined && year2.get >= MIN_YEAR && year2.get <= MAX_YEAR) year2 else None

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

    def addNumberSpan(span: (Int, Int), number: Float): Unit = {
      if (!spans.contains(span))
        spans(span) = Array()
      spans(span) = spans(span) ++ Array(TapasCellValue(Some(number), TapasCellDate.Empty()))
    }

    def addDateSpan(span: (Int, Int), date: TapasCellDate): Unit = {
      if (!spans.contains(span))
        spans(span) = Array()
      spans(span) = spans(span) ++ Array(TapasCellValue(None, date))
    }

    // add numbers using pattern
    NUMBER_PATTERN
      .findAllMatchIn(text)
      .foreach(m => {
        val spanText = text.slice(m.start, m.end)
        val number = parseNumber(spanText)
        if (number.isDefined)
          addNumberSpan((m.start, m.end), number.get)
      })

    // add numbers
    getAllSpans(text, 1)
      .filter(span => !spans.contains(span))
      .foreach(span => {
        val spanText = text.slice(span._1, span._2)
        val number = parseNumber(spanText)
        if (number.isDefined)
          addNumberSpan(span, number.get)
        NUMBER_WORDS.zipWithIndex.foreach { case (numWord, index) =>
          if (spanText == numWord)
            addNumberSpan(span, index.toFloat)
        }
        ORDINAL_WORDS.zipWithIndex.foreach { case (numWord, index) =>
          if (spanText == numWord)
            addNumberSpan(span, index.toFloat)
        }
      })
    // add dates
    getAllSpans(text, 5).foreach(span => {
      val spanText = text.slice(span._1, span._2)
      val date = parseDate(spanText)
      if (date.isDefined)
        addDateSpan(span, date.get)
    })

    val sortedSpans = spans.toArray.sortBy(x => (x._1._2 - x._1._1, -x._1._1)).reverse
    var selectedSpans = collection.mutable.ArrayBuffer[((Int, Int), Array[TapasCellValue])]()
    sortedSpans.foreach { case (span, values) =>
      if (!selectedSpans
          .map(_._1)
          .exists(selectedSpan => selectedSpan._1 <= span._1 && span._2 <= selectedSpan._2)) {
        selectedSpans = selectedSpans ++ Array((span, values))
      }
    }
    selectedSpans
      .sortBy(x => x._1._1)
      .flatMap { case (span, values) =>
        values.map(value => TapasNumericValueSpan(span._1, span._2, value))
      }
      .toArray
  }

  def encodeTapasData(
      questions: Seq[String],
      table: TableData,
      caseSensitive: Boolean,
      maxSentenceLength: Int): Seq[TapasInputData] = {

    val basicTokenizer = new BasicTokenizer(caseSensitive = true, hasBeginEnd = false)

    val questionInputIds = questions.map(question => {

      val sentence = new Sentence(start = 0, end = question.length, content = question, index = 0)
      val tokens = basicTokenizer.tokenize(sentence)
      (if (caseSensitive)
         tokens
       else
         tokens.map(x => IndexedToken(x.token.toLowerCase(), x.begin, x.end)))
        .flatMap(token => encoder.encode(token))
        .map(_.pieceId)
    })
    val maxQuestionLength = questionInputIds.map(_.length).max

    val inputIds = collection.mutable.ArrayBuffer[Int]()
    val attentionMask = collection.mutable.ArrayBuffer[Int]()
    val segmentIds = collection.mutable.ArrayBuffer[Int]()
    val columnIds = collection.mutable.ArrayBuffer[Int]()
    val rowIds = collection.mutable.ArrayBuffer[Int]()
    val prevLabels = collection.mutable.ArrayBuffer[Int]()
    val columnRanks = collection.mutable.ArrayBuffer[Int]()
    val invertedColumnRanks = collection.mutable.ArrayBuffer[Int]()

    table.header.indices.foreach(colIndex => {
      val sentence = new Sentence(
        start = 0,
        end = table.header(colIndex).length,
        content = table.header(colIndex),
        index = colIndex)
      val tokens = basicTokenizer.tokenize(sentence)
      val columnInputIds =
        if (caseSensitive)
          tokens
        else
          tokens.map(x => IndexedToken(x.token.toLowerCase(), x.begin, x.end))

      columnInputIds
        .flatMap(token => encoder.encode(token))
        .foreach(x => {
          inputIds.append(x.pieceId)
          attentionMask.append(1)
          segmentIds.append(1)
          columnIds.append(colIndex + 1)
          rowIds.append(0)
          prevLabels.append(0)
          columnRanks.append(0)
          invertedColumnRanks.append(0)
        })

    })

    val tableCellValues =
      collection.mutable.Map[Int, Array[(TapasNumericValueSpan, Int, Array[Int])]]()

    table.rows.indices
      .map(rowIndex => {
        table.header.indices
          .map(colIndex => {
            val cellText = table.rows(rowIndex)(colIndex)
            val sentence =
              new Sentence(start = 0, end = cellText.length, content = cellText, index = 0)
            val tokens = basicTokenizer.tokenize(sentence)
            val cellInputIds = (if (caseSensitive)
                                  tokens
                                else
                                  tokens.map(x =>
                                    IndexedToken(x.token.toLowerCase(), x.begin, x.end)))
              .flatMap(token => encoder.encode(token))

            cellInputIds.foreach(x => {
              inputIds.append(x.pieceId)
              attentionMask.append(1)
              segmentIds.append(1)
              columnIds.append(colIndex + 1)
              rowIds.append(rowIndex + 1)
              prevLabels.append(0)
              columnRanks.append(0)
              invertedColumnRanks.append(0)
            })

            val tapasNumValuesWithTokenIndices = parseText(cellText).map(numValue =>
              (
                numValue,
                rowIndex,
                cellInputIds.zipWithIndex
                  //                                  .filter(id => id._1.begin>=numValue.begin && id._1.end <= numValue.end)
                  .map(_._2 + (inputIds.length - cellInputIds.length))))
            tableCellValues(colIndex) =
              tableCellValues.getOrElse(colIndex, Array()) ++ tapasNumValuesWithTokenIndices

          })
          .toArray
      })
      .toArray

    // Compute column ranks
    tableCellValues.foreach { case (_, values) =>
      val rowsWithNumberValues =
        values.filter(x => x._1.value.number.isDefined).map(_._2).distinct.length
      val rowsWithDateValues =
        values.filter(x => x._1.value.number.isEmpty).map(_._2).distinct.length

      val sortedValues =
        if (rowsWithNumberValues >= math.max(
            rowsWithDateValues,
            MIN_NUMBER_OF_ROWS_WITH_VALUES_PROPORTION * table.rows.length)) {
          values.sortWith((v1, v2) => TapasCellValue.compareNumbers(v1._1.value, v2._1.value))
        } else if (rowsWithDateValues >= math.max(
            rowsWithNumberValues,
            MIN_NUMBER_OF_ROWS_WITH_VALUES_PROPORTION * table.rows.length)) {
          values.sortWith((v1, v2) => TapasCellValue.compareDates(v1._1.value, v2._1.value))
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

    // generate data
    questions.zip(questionInputIds).map { case (question, qIds) =>
      // compute numeric relations
      val numericRelations = collection.mutable.ArrayBuffer.fill(inputIds.length)(0)
      val questionNumValues = parseText(question)
      val cellRelations = collection.mutable.Map[(Int, Int), Array[Int]]()
      questionNumValues.foreach(questionNumValue => {
        tableCellValues.foreach { case (columnIdx, cellValues) =>
          cellValues
            .foreach(cellValue => {
              val rel = questionNumValue.value.getNumericRelationTo(cellValue._1.value)
              if (rel != TapasNumericRelation.NONE)
                cellRelations((columnIdx, cellValue._2)) = cellRelations
                  .getOrElse((columnIdx, cellValue._2), Array()) ++ Array(rel)
            })
        }
      })
      cellRelations.foreach { case (pos, relations) =>
        tableCellValues(pos._1)
          .filter(_._2 == pos._2)
          .flatMap(_._3)
          .foreach(tokenIndex =>
            numericRelations(tokenIndex) = relations
              .map(rel => math.pow(2, rel - TapasNumericRelation.EQ))
              .sum
              .toInt)
      }
      val emptyTokenTypes = Array.fill(qIds.length + 2)(0)
      val padding = Array.fill(maxQuestionLength - qIds.length)(0)

      def setMaxSentenceLimit(vector: Array[Int]): Array[Int] = {
        vector.slice(0, math.min(maxSentenceLength, vector.length))
      }

      TapasInputData(
        inputIds = setMaxSentenceLimit(
          Array(sentenceStartTokenId) ++ qIds ++ Array(
            sentenceEndTokenId) ++ inputIds ++ padding),
        attentionMask =
          setMaxSentenceLimit(emptyTokenTypes.map(_ => 1) ++ attentionMask ++ padding),
        segmentIds = setMaxSentenceLimit(emptyTokenTypes ++ segmentIds ++ padding),
        columnIds = setMaxSentenceLimit(emptyTokenTypes ++ columnIds ++ padding),
        rowIds = setMaxSentenceLimit(emptyTokenTypes ++ rowIds ++ padding),
        prevLabels = setMaxSentenceLimit(emptyTokenTypes ++ prevLabels ++ padding),
        columnRanks = setMaxSentenceLimit(emptyTokenTypes ++ columnRanks ++ padding),
        invertedColumnRanks =
          setMaxSentenceLimit(emptyTokenTypes ++ invertedColumnRanks ++ padding),
        numericRelations = setMaxSentenceLimit(emptyTokenTypes ++ numericRelations ++ padding))
    }

  }
}
