package com.johnsnowlabs.nlp.annotators.tokenizer.moses

import scala.util.matching.Regex

/**
  * Scala Port of the Moses Tokenizer from [[https://github.com/alvations/sacremoses scaremoses]].
  */
private[johnsnowlabs] class MosesTokenizer(lang: String) {
  require(lang == "en", "Only english is supported at the moment.")
  private val DEDUPLICATE_SPACE = (raw"""\s+""".r, " ")
  private val ASCII_JUNK = (raw"""[\000-\037]""".r, "")

  private val IsAlpha = raw"""\p{L}"""
  private val IsN = raw"""\p{N}"""
  private val IsAlnum = IsAlpha + IsN // TODO: Lesser used languages like Tibetan, Khmer, Cham etc.
  private val PAD_NOT_ISALNUM = (raw"""([^$IsAlnum\s\.'\`\,\-])""".r, " $1 ")

  private val COMMA_SEPARATE_1 = (raw"""([^$IsN])[,]""".r, "$1 , ")
  private val COMMA_SEPARATE_2 = (raw"""[,]([^$IsN])""".r, " , $1")
  private val COMMA_SEPARATE_3 = (raw"""([$IsN])[,]$$""".r, "$1 , ")

  private val EN_SPECIFIC_1 = (raw"""([^$IsAlpha])[']([^$IsAlpha])""".r, "$1 ' $2")
  private val EN_SPECIFIC_2 = (raw"""([^$IsAlpha$IsN])[']([$IsAlpha])""".r, "$1 ' $2")
  private val EN_SPECIFIC_3 = (raw"""([$IsAlpha])[']([^$IsAlpha])""".r, "$1 ' $2")
  private val EN_SPECIFIC_4 = (raw"""([$IsAlpha])[']([$IsAlpha])""".r, "$1 '$2")
  private val EN_SPECIFIC_5 = (raw"""([$IsN])[']([s])""".r, "$1 '$2")
  private val ENGLISH_SPECIFIC_APOSTROPHE = Array(
    EN_SPECIFIC_1,
    EN_SPECIFIC_2,
    EN_SPECIFIC_3,
    EN_SPECIFIC_4,
    EN_SPECIFIC_5
  )
  private val NON_SPECIFIC_APOSTROPHE = (raw"""\'""".r, " ' ")
  private val TRAILING_DOT_APOSTROPHE = (raw"""\.' ?$$""".r, " . ' ")
  // TODO: Dynamic from file
  private val NONBREAKING_PREFIXES = Array("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Adj", "Adm", "Adv", "Asst", "Bart", "Bldg", "Brig", "Bros", "Capt", "Cmdr", "Col", "Comdr", "Con", "Corp", "Cpl", "DR", "Dr", "Drs", "Ens", "Gen", "Gov", "Hon", "Hr", "Hosp", "Insp", "Lt", "MM", "MR", "MRS", "MS", "Maj", "Messrs", "Mlle", "Mme", "Mr", "Mrs", "Ms", "Msgr", "Op", "Ord", "Pfc", "Ph", "Prof", "Pvt", "Rep", "Reps", "Res", "Rev", "Rt", "Sen", "Sens", "Sfc", "Sgt", "Sr", "St", "Supt", "Surg", "v", "vs", "i.e", "rev", "e.g", "No #NUMERIC_ONLY#", "Nos", "Art #NUMERIC_ONLY#", "Nr", "pp #NUMERIC_ONLY#", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
  private val NUMERIC_ONLY_PREFIXES = Array("No", "Art", "pp")

  def applySubstitution(text: String, patternReplacements: (Regex, String)*): String = {
    var processed = text
    for ((pattern, sub) <- patternReplacements) {
      processed = pattern.replaceAllIn(processed, sub)
      //        processed = processed.replaceAll(pattern, sub)
    }
    processed
  }

  private val MULTIDOT = (raw"""\.([\.]+)""".r, " DOTMULTI$1")
  private val MULTIDOT_SUB_1 = (raw"""DOTMULTI\.([^\.])""".r, "DOTDOTMULTI $1")
  private val MULTIDOT_SUB_2 = (raw"""DOTMULTI\.""".r, "DOTDOTMULTI")

  private def replaceMultiDots(text: String): String = {
    var processed: String = text
    processed = applySubstitution(processed, MULTIDOT)
    while (processed.indexOf("DOTMULTI.") >= 0) {
      processed = applySubstitution(processed, MULTIDOT_SUB_1)
      processed = applySubstitution(processed, MULTIDOT_SUB_2)
    }
    processed
  }

  private def isAnyAlpha(s: String): Boolean = s"[$IsAlnum]".r.findFirstIn(s) match {
    case Some(_) => true
    case None => false
  }

  private def isLower(s: String): Boolean = s.forall(_.isLower) // TODO Some languages missing

  private val IS_NUMERIC_ONLY = raw"""^[0-9]+""".r
  private val ENDS_WITH_PERIOD = raw"""^(\S+)\.$$""".r

  def handlesNonBreakingPrefixes(text: String): String = {
    // Splits the text into tokens to check for nonbreaking prefixes.
    val tokens = text.split(" ")
    val numTokens = tokens.length
    for ((token, i) <- tokens.zipWithIndex) {
      // Checks if token ends with a full stop
      val tokenEndsWithPeriod = ENDS_WITH_PERIOD.findFirstMatchIn(token)
      tokenEndsWithPeriod match {
        case None => tokenEndsWithPeriod
        case Some(prefixMatch) =>
          val prefix = prefixMatch.group(1)

          // Checks for 3 conditions if
          // i.   the prefix contains a fullstop and
          //      any char in the prefix is within the IsAlpha charset
          // ii.  the prefix is in the list of nonbreaking prefixes and
          //      does not contain #NUMERIC_ONLY#
          // iii. the token is not the last token and that the
          //      next token contains all lowercase.

          // No change to the token.
          // Checks if the prefix is in NUMERIC_ONLY_PREFIXES
          // and ensures that the next word is a digit.
          def containsFullStopAndIsAlpha = ((prefix contains ".") && isAnyAlpha(prefix)) ||
            (NONBREAKING_PREFIXES.contains(prefix) && !NUMERIC_ONLY_PREFIXES.contains(prefix)) ||
            (
              (i != numTokens - 1)
                && tokens(i + 1).nonEmpty
                && isLower(tokens(i + 1)(0).toString)
              )

          // No change to the token.
          def isNonBreakingAndNumericOnly = {
            (
              NONBREAKING_PREFIXES.contains(prefix)
                && ((i + 1) < numTokens)
                && IS_NUMERIC_ONLY.findFirstIn(tokens(i + 1)).isDefined
              )
          }
          // Otherwise, adds a space after the tokens before a dot.
          if (!containsFullStopAndIsAlpha && !isNonBreakingAndNumericOnly) tokens(i) = prefix + " ."
      }
    }
    tokens.mkString(" ") // Stitch the tokens back.
  }

  private val RESTORE_MULTIDOT_1 = ("DOTDOTMULTI".r, "DOTMULTI.")
  private val RESTORE_MULTIDOT_2 = ("DOTMULTI".r, ".")

  private def restoreMultiDots(text: String) = {
    var processed = text
    while (processed.indexOf("DOTDOTMULTI") >= 0) { // re.search(r"DOTDOTMULTI", text):
      processed = applySubstitution(processed, RESTORE_MULTIDOT_1)
    }
    applySubstitution(processed, RESTORE_MULTIDOT_2)
  }

  def tokenize(text: String): Array[String] = {
    var processed = text

    processed = applySubstitution(processed, DEDUPLICATE_SPACE, ASCII_JUNK)
    processed = processed.trim()

    //    if (protectedPatterns) ???

    processed = applySubstitution(processed, PAD_NOT_ISALNUM)

    //    if (aggressiveDashSplits) ???

    processed = replaceMultiDots(processed)

    processed = applySubstitution(processed, COMMA_SEPARATE_1, COMMA_SEPARATE_2, COMMA_SEPARATE_3)

    if (lang == "en") processed = applySubstitution(processed, ENGLISH_SPECIFIC_APOSTROPHE: _*)
    else if (lang == "it" || lang == "fr") ??? // TODO
    else processed = applySubstitution(processed, NON_SPECIFIC_APOSTROPHE)

    processed = handlesNonBreakingPrefixes(processed)

    processed = applySubstitution(processed, DEDUPLICATE_SPACE).trim()

    processed = applySubstitution(processed, TRAILING_DOT_APOSTROPHE)

    // Restore the protected tokens.
    // if (protectedPatterns) ???

    processed = restoreMultiDots(processed)
    processed.split(" ")
  }
}