package com.johnsnowlabs.util


case class Version(parts: List[Int]) {
  override def toString(): String = {
    parts.mkString(".")
  }

  def take(n: Int): Version = {
    Version(parts.take(n))
  }
}

object Version {
  def apply(parts: Int*): Version = Version(parts.toList)

  def isInteger(str: String) = str.nonEmpty && str.forall(c => Character.isDigit(c))

  def parse(str: String): Version = {
    val parts = str.split('.')
      .takeWhile(p => isInteger(p))
      .map(p => p.toInt)
      .toList

    Version(parts)
  }

  def isCompatible(current: Version, found: Version): Boolean = isCompatible(current, Some(found))

  /**
    * Checks weather found version could be used with current version
    * @param current Version of current library
    * @param found Version of library of found resource
    * @return True ar False
    *
    * Examples (current) and (found):
    * 1.2.3 and 1.2   => True
    * 1.2   and 1.2.3 => False (found more specific version)    *
    * 1.2   and None  => True  (found version that could be used with all versions)
    */
  def isCompatible(current: Version, found: Option[Version]): Boolean = {
    found.forall{f =>
      val cParts = current.parts
      val fParts = f.parts

      // If found version more specific than ours then we can't use it
      if (cParts.length < fParts.length)
        false
      else {
        // All first digits must be equals:
        var previousWasBigger: Option[Boolean] = None
        cParts.zip(fParts).forall{case(c, t) =>
          if (previousWasBigger.exists(v => v) || t == c)
            true
          else if (c > t && previousWasBigger.isEmpty) {
            previousWasBigger = Some(true)
            true
          }
          else {
            previousWasBigger = Some(false)
            false
          }
        }
      }
    }
  }
}