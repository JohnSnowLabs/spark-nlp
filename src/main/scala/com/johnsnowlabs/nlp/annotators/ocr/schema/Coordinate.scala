package com.johnsnowlabs.nlp.annotators.ocr.schema

/**
  *
  * @param i  Chunk index.
  * @param p  Page number.
  * @param x  The lower left x coordinate.
  * @param y  The lower left y coordinate.
  * @param w  The width of the rectangle.
  * @param h  The height of the rectangle.
  */
case class Coordinate(i: Int, p: Int, x: Float, y: Float, w: Float, h: Float)
