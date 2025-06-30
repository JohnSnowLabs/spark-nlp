package com.johnsnowlabs.partition

import com.johnsnowlabs.reader.HTMLElement

case class Chunk(elements: List[HTMLElement]) {
  def length: Int = elements.map(_.content.length).sum
}
