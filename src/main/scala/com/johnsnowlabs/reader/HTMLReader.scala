/*
 * Copyright 2017-2024 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.{isValidURL, validFile}
import com.johnsnowlabs.partition.util.PartitionHelper.datasetWithTextFile
import com.johnsnowlabs.reader.util.HTMLParser
import com.johnsnowlabs.reader.util.HTMLParser.tableElementToJson
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.jsoup.Jsoup
import org.jsoup.nodes.{Document, Element, Node, TextNode}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/** Class to parse and read HTML files.
  *
  * @param titleFontSize
  *   Minimum font size threshold in pixels used as part of heuristic rules to detect title
  *   elements based on formatting (e.g., bold, centered, capitalized). By default, it is set to
  *   16.
  * @param storeContent
  *   Whether to include the raw file content in the output DataFrame as a separate 'content'
  *   column, alongside the structured output. By default, it is set to false.
  * @param timeout
  *   Timeout value in seconds for reading remote HTML resources. Applied when fetching content
  *   from URLs. By default, it is set to 0.
  * @param headers
  *   sets the necessary headers for the URL request.
  *
  * Two types of input paths are supported for the reader,
  *
  * htmlPath: this is a path to a directory of HTML files or a path to an HTML file E.g.
  * "path/html/files"
  *
  * url: this is the URL or set of URLs of a website . E.g., "https://www.wikipedia.org"
  *
  * ==Example==
  * {{{
  * val path = "./html-files/fake-html.html"
  * val HTMLReader = new HTMLReader()
  * val htmlDF = HTMLReader.read(url)
  * }}}
  *
  * {{{
  * htmlDF.show()
  * +--------------------+--------------------+
  * |                path|                html|
  * +--------------------+--------------------+
  * |file:/content/htm...|[{Title, My First...|
  * +--------------------+--------------------+
  *
  * htmlDf.printSchema()
  * root
  *  |-- path: string (nullable = true)
  *  |-- html: array (nullable = true)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- elementType: string (nullable = true)
  *  |    |    |-- content: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  * }}}
  * For more examples please refer to this
  * [[https://github.com/JohnSnowLabs/spark-nlp/examples/python/reader/SparkNLP_HTML_Reader_Demo.ipynb notebook]].
  */

class HTMLReader(
    titleFontSize: Int = 16,
    storeContent: Boolean = false,
    timeout: Int = 0,
    includeTitleTag: Boolean = false,
    outputFormat: String = "plain-text",
    headers: Map[String, String] = Map.empty)
    extends Serializable {

  private lazy val spark = ResourceHelper.spark
  import spark.implicits._

  private var outputColumn = "html"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def getOutputColumn: String = outputColumn

  /** @param inputSource
    *   this is the link to the URL E.g. www.wikipedia.com
    *
    * @return
    *   Dataframe with parsed URL content.
    */

  def read(inputSource: String): DataFrame = {
    ResourceHelper match {
      case _ if validFile(inputSource) && !inputSource.startsWith("http") =>
        val htmlDf = datasetWithTextFile(spark, inputSource)
          .withColumn(outputColumn, parseHtmlUDF(col("content")))
        if (storeContent) htmlDf.select("path", "content", outputColumn)
        else htmlDf.select("path", outputColumn)
      case _ if isValidURL(inputSource) =>
        val htmlDf = spark
          .createDataset(Seq(inputSource))
          .toDF("url")
          .withColumn(outputColumn, parseURLUDF(col("url")))
        if (storeContent) htmlDf.select("url", "content", outputColumn)
        else htmlDf.select("url", outputColumn)
      case _ =>
        throw new IllegalArgumentException(s"Invalid inputSource: $inputSource")
    }
  }

  /** @param inputURLs
    *   this is a list of URLs E.g. [www.wikipedia.com, www.example.com]
    *
    * @return
    *   Dataframe with parsed URL content.
    */

  def read(inputURLs: Array[String]): DataFrame = {
    val spark = ResourceHelper.spark
    import spark.implicits._

    val validURLs = inputURLs.filter(url => ResourceHelper.isValidURL(url)).toSeq
    spark
      .createDataset(validURLs)
      .toDF("url")
      .withColumn(outputColumn, parseURLUDF(col("url")))
  }

  private val parseHtmlUDF = udf((html: String) => {
    val document = Jsoup.parse(html)
    startTraversalFromBody(document)
  })

  private val parseURLUDF = udf((url: String) => {
    val connection = Jsoup
      .connect(url)
      .headers(headers.asJava)
      .timeout(timeout * 1000)
    val document = connection.get()
    startTraversalFromBody(document)
  })

  private def startTraversalFromBody(document: Document): Array[HTMLElement] = {
    val body = document.body()
    val elements = extractElements(body)
    val docTitle = document.title().trim

    if (docTitle.nonEmpty && includeTitleTag) {
      val titleElem = HTMLElement(
        ElementType.TITLE,
        content = docTitle,
        metadata = mutable.Map.empty[String, String])
      Array(titleElem) ++ elements
    } else {
      elements
    }
  }

  def htmlToHTMLElement(html: String): Array[HTMLElement] = {
    val document = Jsoup.parse(html)
    startTraversalFromBody(document)
  }

  def urlToHTMLElement(url: String): Array[HTMLElement] = {
    val connection = Jsoup
      .connect(url)
      .headers(headers.asJava)
      .timeout(timeout * 1000)
    val document = connection.get()
    startTraversalFromBody(document)
  }

  private case class NodeMetadata(tagName: Option[String], hidden: Boolean, var visited: Boolean)

  private def extractElements(root: Node): Array[HTMLElement] = {
    var sentenceIndex = 0
    val elements = ArrayBuffer[HTMLElement]()
    val trackingNodes = mutable.Map[Node, NodeMetadata]()
    var pageNumber = 1

    def isNodeHidden(node: Node): Boolean = {
      node match {
        case elem: Element =>
          val style = elem.attr("style").toLowerCase
          val isHiddenByStyle =
            style.contains("display:none") || style.contains("visibility:hidden")
          val isHiddenByAttribute = elem.hasAttr("hidden") || elem.attr("aria-hidden") == "true"
          isHiddenByStyle || isHiddenByAttribute
        case _ => false
      }
    }

    def collectTextFromNodes(nodes: List[Node]): String = {
      val textBuffer = ArrayBuffer[String]()

      def traverseAndCollect(node: Node): Unit = {
        val isHiddenNode = trackingNodes
          .getOrElseUpdate(
            node,
            NodeMetadata(tagName = getTagName(node), hidden = isNodeHidden(node), visited = true))
          .hidden
        if (!isHiddenNode) {
          node match {
            case textNode: TextNode =>
              trackingNodes(textNode).visited = true
              val text = textNode.text().trim
              if (text.nonEmpty) textBuffer += text

            case elem: Element =>
              trackingNodes(elem).visited = true
              val text = elem.ownText().trim
              if (text.nonEmpty) textBuffer += text
              // Recursively collect text from all child nodes
              elem.childNodes().asScala.foreach(traverseAndCollect)

            case _ => // Ignore other node types
          }
        }
      }

      // Start traversal for each node in the list
      nodes.foreach(traverseAndCollect)
      textBuffer.mkString(" ").replaceAll("\\s+", " ").trim
    }

    def traverse(node: Node, tagName: Option[String]): Unit = {
      trackingNodes.getOrElseUpdate(
        node,
        NodeMetadata(tagName = tagName, hidden = isNodeHidden(node), visited = false))

      node.childNodes().forEach { childNode =>
        trackingNodes.getOrElseUpdate(
          childNode,
          NodeMetadata(tagName = tagName, hidden = isNodeHidden(childNode), visited = false))
      }

      if (trackingNodes(node).hidden) {
        return
      }

      node match {
        case element: Element =>
          val visitedNode = trackingNodes(element).visited
          val pageMetadata: mutable.Map[String, String] =
            mutable.Map("pageNumber" -> pageNumber.toString)

          element.tagName() match {
            case "a" =>
              pageMetadata("sentence") = sentenceIndex.toString
              sentenceIndex += 1
              val href = element.attr("href").trim
              val linkText = element.text().trim
              if (href.nonEmpty && linkText.nonEmpty && !visitedNode) {
                trackingNodes(element).visited = true
                elements += HTMLElement(
                  ElementType.LINK,
                  content = s"[$linkText]($href)",
                  metadata = pageMetadata)
              }
            case "table" =>
              pageMetadata("sentence") = sentenceIndex.toString
              sentenceIndex += 1
              val tableContent = outputFormat match {
                case "plain-text" =>
                  extractNestedTableContent(element).trim
                case "html-table" =>
                  element
                    .outerHtml()
                    .replaceAll("\\n", "")
                    .replaceAll(">\\s+<", "><")
                    .replaceAll("^\\s+|\\s+$", "")
                case "json-table" =>
                  tableElementToJson(element)
                case _ =>
                  extractNestedTableContent(element).trim
              }
              if (tableContent.nonEmpty && !visitedNode) {
                trackingNodes(element).visited = true
                elements += HTMLElement(
                  ElementType.TABLE,
                  content = tableContent,
                  metadata = pageMetadata)
              }
            case "li" =>
              pageMetadata("sentence") = sentenceIndex.toString
              sentenceIndex += 1
              val itemText = element.text().trim
              if (itemText.nonEmpty && !visitedNode) {
                trackingNodes(element).visited = true
                elements += HTMLElement(
                  ElementType.LIST_ITEM,
                  content = itemText,
                  metadata = pageMetadata)
              }
            case "pre" =>
              // A <pre> tag typically contains a <code> child
              val codeElem = element.getElementsByTag("code").first()
              val codeText =
                if (codeElem != null) codeElem.text().trim
                else element.text().trim
              if (codeText.nonEmpty && !visitedNode) {
                pageMetadata("sentence") = sentenceIndex.toString
                sentenceIndex += 1
                trackingNodes(element).visited = true
                elements += HTMLElement(
                  ElementType.UNCATEGORIZED_TEXT,
                  content = codeText,
                  metadata = pageMetadata)
              }
            case tag if isParagraphLikeElement(element) =>
              if (!visitedNode) {
                val classType = classifyParagraphElement(element)

                // Traverse children first so that <img>, <a>, etc. inside the paragraph are processed
                element.childNodes().asScala.foreach { childNode =>
                  val tagName = getTagName(childNode)
                  traverse(childNode, tagName)
                }

                // Now handle the paragraph itself
                classType match {
                  case ElementType.NARRATIVE_TEXT =>
                    val childNodes = element.childNodes().asScala.toList
                    val aggregatedText = collectTextFromNodes(childNodes)
                    if (aggregatedText.nonEmpty) {
                      pageMetadata("sentence") = sentenceIndex.toString
                      sentenceIndex += 1
                      trackingNodes(element).visited = true
                      elements += HTMLElement(
                        ElementType.NARRATIVE_TEXT,
                        content = aggregatedText,
                        metadata = pageMetadata)
                    }

                  case ElementType.TITLE =>
                    val titleText = element.text().trim
                    if (titleText.nonEmpty) {
                      pageMetadata("sentence") = sentenceIndex.toString
                      sentenceIndex += 1
                      trackingNodes(element).visited = true
                      elements += HTMLElement(
                        ElementType.TITLE,
                        content = titleText,
                        metadata = pageMetadata)
                    }

                  case ElementType.UNCATEGORIZED_TEXT =>
                    val text = element.text().trim
                    if (text.nonEmpty) {
                      pageMetadata("sentence") = sentenceIndex.toString
                      sentenceIndex += 1
                      trackingNodes(element).visited = true
                      elements += HTMLElement(
                        ElementType.UNCATEGORIZED_TEXT,
                        content = text,
                        metadata = pageMetadata)
                    }
                }
              }
            case _ if isTitleElement(element) && !visitedNode =>
              trackingNodes(element).visited = true
              val titleText = element.text().trim
              if (titleText.nonEmpty) {
                pageMetadata("sentence") = sentenceIndex.toString
                sentenceIndex += 1
                elements += HTMLElement(
                  ElementType.TITLE,
                  content = titleText,
                  metadata = pageMetadata)
              }
            case "hr" =>
              if (element.attr("style").toLowerCase.contains("page-break")) {
                pageNumber = pageNumber + 1
              }
            case "img" =>
              pageMetadata("sentence") = sentenceIndex.toString
              sentenceIndex += 1
              val src = element.attr("src").trim
              val alt = element.attr("alt").trim
              if (src.nonEmpty && !visitedNode) {
                trackingNodes(element).visited = true
                val isBase64 = src.toLowerCase.contains("base64")
                val width = element.attr("width").trim
                val height = element.attr("height").trim

                val imgMetadata = mutable.Map[String, String]("alt" -> alt) ++ pageMetadata

                var contentValue = src
                if (isBase64) {
                  val commaIndex = src.indexOf(',')
                  if (commaIndex > 0) {
                    val header = src.substring(0, commaIndex)
                    val base64Payload = src.substring(commaIndex + 1)
                    imgMetadata("encoding") = header
                    contentValue = base64Payload
                  }
                }

                if (width.nonEmpty) imgMetadata("width") = width
                if (height.nonEmpty) imgMetadata("height") = height
                elements += HTMLElement(
                  ElementType.IMAGE,
                  content = contentValue,
                  metadata = imgMetadata)
              }
            case _ =>
              element.childNodes().asScala.foreach { childNode =>
                val tagName = getTagName(childNode)
                traverse(childNode, tagName)
              }
          }
        case _ => // Ignore other node types
      }
    }

    // Start traversal from the root node
    val tagName = getTagName(root)
    traverse(root, tagName)
    elements.toArray
  }

  private def isParagraphLikeElement(elem: Element): Boolean = {
    val tag = elem.tagName().toLowerCase
    val style = elem.attr("style").toLowerCase
    (tag == "p") ||
    (tag == "div" && (
      style.contains("font-size") ||
        style.contains("line-height") ||
        style.contains("margin") ||
        elem.getElementsByTag("b").size() > 0 ||
        elem.getElementsByTag("strong").size() > 0
    ))
  }

  private def getTagName(node: Node): Option[String] = {
    node match {
      case element: Element => Some(element.tagName())
      case _ => None
    }
  }

  private def classifyParagraphElement(element: Element): String = {
    if (isFormattedAsTitle(element)) {
      ElementType.TITLE
    } else if (isTextElement(element)) {
      ElementType.NARRATIVE_TEXT
    } else {
      ElementType.UNCATEGORIZED_TEXT
    }
  }

  private def isTitleElement(element: Element): Boolean = {
    val tag = element.tagName().toLowerCase
    val style = element.attr("style").toLowerCase
    val role = element.attr("role").toLowerCase
    HTMLParser.isTitleElement(tag, style, role, titleFontSize)
  }

  private def isTextElement(elem: Element): Boolean = {
    !isFormattedAsTitle(elem) &&
    (elem.attr("style").toLowerCase.contains("text") ||
      elem.tagName().toLowerCase == "p" ||
      (elem.tagName().toLowerCase == "div" && isParagraphLikeElement(elem)))
  }

  private def isFormattedAsTitle(elem: Element): Boolean = {
    val style = elem.attr("style").toLowerCase
    val hasBoldTag =
      elem.getElementsByTag("b").size() > 0 || elem.getElementsByTag("strong").size() > 0
    hasBoldTag || HTMLParser.isFormattedAsTitle(style, titleFontSize)
  }

  private def extractNestedTableContent(elem: Element): String = {
    val textBuffer = ArrayBuffer[String]()
    val processedElements = mutable.Set[Node]() // Set to track processed elements

    // Recursive function to collect text from the element and its children
    def collectText(node: Node): Unit = {
      node match {
        case childElem: Element =>
          if (!processedElements.contains(childElem)) {
            processedElements += childElem

            val directText = childElem.ownText().trim
            if (directText.nonEmpty) textBuffer += directText

            childElem.childNodes().asScala.foreach(collectText)
          }

        case _ => // Ignore other node types
      }
    }

    // Start the recursive text collection
    collectText(elem)
    textBuffer.mkString(" ")
  }

}
