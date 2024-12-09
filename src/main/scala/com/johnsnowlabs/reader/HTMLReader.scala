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
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.jsoup.Jsoup
import org.jsoup.nodes.{Document, Element, Node, TextNode}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class HTMLReader(titleFontSize: Int = 16) extends Serializable {

  private val spark = ResourceHelper.spark
  import spark.implicits._

  def read(inputSource: String): DataFrame = {

    ResourceHelper match {
      case _ if validFile(inputSource) && !inputSource.startsWith("http") =>
        spark.sparkContext
          .wholeTextFiles(inputSource)
          .toDF("path", "content")
          .withColumn("html", parseHtmlUDF(col("content")))

      case _ if isValidURL(inputSource) =>
        spark
          .createDataset(Seq(inputSource))
          .toDF("url")
          .withColumn("html", parseURLUDF(col("url")))
      case _ =>
        throw new IllegalArgumentException(s"Invalid inputSource: $inputSource")
    }
  }

  def read(inputURLs: Array[String]): DataFrame = {
    val spark = ResourceHelper.spark
    import spark.implicits._

    val validURLs = inputURLs.filter(url => ResourceHelper.isValidURL(url)).toSeq
    spark
      .createDataset(validURLs)
      .toDF("url")
      .withColumn("html", parseURLUDF(col("url")))
  }

  private val parseHtmlUDF = udf((html: String) => {
    val document = Jsoup.parse(html)
    startTraversalFromBody(document)
  })

  private val parseURLUDF = udf((url: String) => {
    val document = Jsoup.connect(url).get()
    startTraversalFromBody(document)
  })

  private def startTraversalFromBody(document: Document): Array[HTMLElement] = {
    val body = document.body()
    extractElements(body)
  }

  private case class NodeMetadata(tagName: Option[String], hidden: Boolean, var visited: Boolean)

  private def extractElements(root: Node): Array[HTMLElement] = {
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
              val tableText = extractNestedTableContent(element).trim
              if (tableText.nonEmpty && !visitedNode) {
                trackingNodes(element).visited = true
                elements += HTMLElement(
                  ElementType.TABLE,
                  content = tableText,
                  metadata = pageMetadata)
              }
            case "p" =>
              if (!visitedNode) {
                classifyParagraphElement(element) match {
                  case ElementType.NARRATIVE_TEXT =>
                    trackingNodes(element).visited = true
                    val childNodes = element.childNodes().asScala.toList
                    val aggregatedText = collectTextFromNodes(childNodes)
                    if (aggregatedText.nonEmpty) {
                      elements += HTMLElement(
                        ElementType.NARRATIVE_TEXT,
                        content = aggregatedText,
                        metadata = pageMetadata)
                    }
                  case ElementType.TITLE =>
                    trackingNodes(element).visited = true
                    val titleText = element.text().trim
                    if (titleText.nonEmpty) {
                      elements += HTMLElement(
                        ElementType.TITLE,
                        content = titleText,
                        metadata = pageMetadata)
                    }
                  case ElementType.UNCATEGORIZED_TEXT =>
                    trackingNodes(element).visited = true
                    val titleText = element.text().trim
                    if (titleText.nonEmpty) {
                      elements += HTMLElement(
                        ElementType.UNCATEGORIZED_TEXT,
                        content = titleText,
                        metadata = pageMetadata)
                    }
                }
              }
            case _ if isTitleElement(element) && !visitedNode =>
              trackingNodes(element).visited = true
              val titleText = element.text().trim
              if (titleText.nonEmpty) {
                elements += HTMLElement(
                  ElementType.TITLE,
                  content = titleText,
                  metadata = pageMetadata)
              }
            case "hr" =>
              if (element.attr("style").toLowerCase.contains("page-break")) {
                pageNumber = pageNumber + 1
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

  private def getTagName(node: Node): Option[String] = {
    node match {
      case element: Element => Some(element.tagName())
      case _ => None
    }
  }

  private def classifyParagraphElement(element: Element): String = {
    if (isTitleElement(element)) {
      ElementType.TITLE
    } else if (isTextElement(element)) {
      ElementType.NARRATIVE_TEXT
    } else {
      ElementType.UNCATEGORIZED_TEXT
    }
  }

  private def isTextElement(elem: Element): Boolean = {
    !isFormattedAsTitle(elem) &&
    (elem.attr("style").toLowerCase.contains("text") || elem.tagName().toLowerCase == "p")
  }

  private def isTitleElement(elem: Element): Boolean = {
    val tag = elem.tagName().toLowerCase

    // Recognize titles from common title-related tags or formatted <p> elements
    tag match {
      case "title" | "h1" | "h2" | "h3" | "header" => true
      case "p" => isFormattedAsTitle(elem) // Check if <p> behaves like a title
      case _ => elem.attr("role").toLowerCase == "heading" // ARIA role="heading"
    }
  }

  private def isFormattedAsTitle(elem: Element): Boolean = {
    // Check for bold text, large font size, or centered alignment
    val style = elem.attr("style").toLowerCase
    val isBold = style.contains("font-weight:bold")
    val isLargeFont = style.contains("font-size") && extractFontSize(style) >= titleFontSize
    val isCentered = style.contains("text-align:center")

    isBold || isLargeFont || (isCentered && isBold) || (isCentered && isLargeFont)
  }

  private def extractFontSize(style: String): Int = {
    val sizePattern = """font-size:(\d+)pt""".r
    sizePattern.findFirstMatchIn(style) match {
      case Some(m) => m.group(1).toInt
      case None => 0
    }
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
