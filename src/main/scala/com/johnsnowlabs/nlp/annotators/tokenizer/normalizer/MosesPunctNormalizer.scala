package com.johnsnowlabs.nlp.annotators.tokenizer.normalizer

private[johnsnowlabs] class MosesPunctNormalizer() {

  private val COMBINED_REPLACEMENT = List(
    (raw"""\r|\u00A0«\u00A0|«\u00A0|«|\u00A0»\u00A0|\u00A0»|»|“|《|》|」|「""", raw""""""),

    (raw""" +""", raw""" """),
    (raw""" +""", raw""" """),
    (raw""" +""", raw""" """),

    (raw"""\( ?""", raw"""("""),

    (raw""" ?\)""", raw""")"""),

    (raw""" :|\u00A0:|∶|：""", raw""":"""),

    (raw""" ;""|\u00A0;|；""", raw""";"""),

    (raw"""`|´|‘|‚|’|’""", raw"""'"""),

    (raw"""''|„|“|”|''|´´""", raw""" " """)
  )
  private val EXTRA_WHITESPACE = List(
    (raw"""\(""", raw""" ("""),
    (raw"""\)""", raw""") """),
    (raw"""\) ([.!:?;,])""", raw""")$$1"""),
    (raw"""(\d) %""", raw"""$$1%""")
  )

  private val NORMALIZE_UNICODE_IF_NOT_PENN = List(
  )

  private val NORMALIZE_UNICODE = List(
    (raw"""–""", raw"""-"""),
    (raw"""—""", raw""" - """),
    (raw"""([a-zA-Z])‘([a-zA-Z])""", raw"""$$1'$$2"""),
    (raw"""([a-zA-Z])’([a-zA-Z])""", raw"""$$1'$$2"""),
    (raw"""…"""", raw"""...""")
  )

  private val FRENCH_QUOTES = List(
  )

  private val HANDLE_PSEUDO_SPACES = List(
    (raw"""\u00A0%""", raw"""%"""),
    (raw"""nº\u00A0""", raw"""nº """),
    (raw"""\u00A0ºC""", raw""" ºC"""),
    (raw"""\u00A0cm""", raw""" cm"""),
    (raw"""\u00A0\\?""", raw"""?"""),
    (raw"""\u00A0\\!""", raw"""!"""),
    (raw""""",\u00A0""", raw""""", """)
  )

  private val EN_QUOTATION_FOLLOWED_BY_COMMA = List((raw""""([,.]+)""", raw"""$$1"""))

  private val DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = List(
    (raw""","""", raw"""","""),
    (raw"""(\.+)"(\s*?[^<])""", raw""""$$1$$2""")
  )

  private val DE_ES_CZ_CS_FR = List(
    (raw"""(\\d)\u00A0(\\d)""", raw"""$$1,$$2""")
  )

  private val OTHER = List(
    (raw"""(\\d)\u00A0(\\d)""", raw"""$$1.$$2""")
  )

  private val REPLACE_UNICODE_PUNCTUATION = List(
    (raw"""，""", raw""","""),
    (raw"""。\s*?""", raw""". """),
    (raw"""、""", raw""","""),
    (raw"""？""", raw"""?"""),
    (raw"""！""", raw"""!"""),
    (raw"""０""", raw"""0"""),
    (raw"""１""", raw"""1"""),
    (raw"""２""", raw"""2"""),
    (raw"""３""", raw"""3"""),
    (raw"""４""", raw"""4"""),
    (raw"""５""", raw"""5"""),
    (raw"""６""", raw"""6"""),
    (raw"""７""", raw"""7"""),
    (raw"""８""", raw"""8"""),
    (raw"""９""", raw"""9"""),
    (raw"""．\s*?""", raw""". """),
    (raw"""～""", raw"""~"""),
    (raw"""…""", raw"""..."""),
    (raw"""━""", raw"""-"""),
    (raw"""〈""", raw"""<"""),
    (raw"""〉""", raw""">"""),
    (raw"""【""", raw"""["""),
    (raw"""】""", raw"""]"""),
    (raw"""％""", raw"""%""")
  )

  private val substitutions = List.concat(
    COMBINED_REPLACEMENT,
    EXTRA_WHITESPACE,
    NORMALIZE_UNICODE_IF_NOT_PENN,
    NORMALIZE_UNICODE,
    FRENCH_QUOTES,
    HANDLE_PSEUDO_SPACES,
    EN_QUOTATION_FOLLOWED_BY_COMMA,
    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA,
    DE_ES_CZ_CS_FR,
    OTHER,
    REPLACE_UNICODE_PUNCTUATION)

  def normalize(text: String): String = {
    var acc = text

    substitutions
      .foreach {
        case (pattern, replacement) =>
          acc = acc.replaceAll(pattern, replacement)
        //          acc = s"$pattern".r.replaceAllIn(acc, replacement)
      }
    acc
  }

  private val printingCharTypes = Set(
    Character.CONTROL,
    Character.DIRECTIONALITY_COMMON_NUMBER_SEPARATOR,
    Character.FORMAT,
    Character.PRIVATE_USE,
    Character.SURROGATE,
    Character.UNASSIGNED
  )

  //  Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
  def removeNonPrintingChar(t: String): String = {
    def isNonPrintingChar(c: Char): Boolean = !printingCharTypes.contains(Character.getType(c).toByte)
    t.toCharArray.filter(isNonPrintingChar).mkString
  }
}
