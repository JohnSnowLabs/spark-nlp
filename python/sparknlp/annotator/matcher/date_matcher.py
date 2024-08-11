#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Contains classes for the DateMatcher."""

from sparknlp.common import *


class DateMatcherUtils(Params):
    """Base class for DateMatcher Annotators
    """
    inputFormats = Param(Params._dummy(),
                         "inputFormats",
                         "input formats list of patterns to match",
                         typeConverter=TypeConverters.toListString)

    outputFormat = Param(Params._dummy(),
                         "outputFormat",
                         "desired output format for dates extracted",
                         typeConverter=TypeConverters.toString)

    readMonthFirst = Param(Params._dummy(),
                           "readMonthFirst",
                           "Whether to parse july 07/05/2015 or as 05/07/2015",
                           typeConverter=TypeConverters.toBoolean
                           )

    defaultDayWhenMissing = Param(Params._dummy(),
                                  "defaultDayWhenMissing",
                                  "which day to set when it is missing from parsed input",
                                  typeConverter=TypeConverters.toInt
                                  )

    anchorDateYear = Param(Params._dummy(),
                           "anchorDateYear",
                           "Add an anchor year for the relative dates such as a day after tomorrow. If not set it "
                           "will use the current year. Example: 2021",
                           typeConverter=TypeConverters.toInt
                           )

    anchorDateMonth = Param(Params._dummy(),
                            "anchorDateMonth",
                            "Add an anchor month for the relative dates such as a day after tomorrow. If not set it "
                            "will use the current month. Example: 1 which means January",
                            typeConverter=TypeConverters.toInt
                            )

    anchorDateDay = Param(Params._dummy(),
                          "anchorDateDay",
                          "Add an anchor day of the day for the relative dates such as a day after tomorrow. If not "
                          "set it will use the current day. Example: 11",
                          typeConverter=TypeConverters.toInt
                          )

    sourceLanguage = Param(Params._dummy(),
                           "sourceLanguage",
                           "source language for explicit translation",
                           typeConverter=TypeConverters.toString)

    relaxedFactoryStrategy = Param(Params._dummy(),
                                   "relaxedFactoryStrategy",
                                   "Matched Strategy to searches relaxed dates",
                                   typeConverter=TypeConverters.toString)

    aggressiveMatching = Param(Params._dummy(),
                               "aggressiveMatching",
                               "Whether to aggressively attempt to find date matches, even in ambiguous or less common formats",
                               typeConverter=TypeConverters.toBoolean)

    def setInputFormats(self, value):
        """Sets input formats patterns to match in the documents.

        Parameters
        ----------
        value : List[str]
            Input formats regex patterns to match dates in documents
        """
        return self._set(inputFormats=value)

    def setOutputFormat(self, value):
        """Sets desired output format for extracted dates, by default yyyy/MM/dd.

        Not all of the date information needs to be included. For example
        ``"YYYY"`` is also a valid input.

        Parameters
        ----------
        value : str
            Desired output format for dates extracted.
        """
        return self._set(outputFormat=value)

    def setReadMonthFirst(self, value):
        """Sets whether to parse the date in mm/dd/yyyy format instead of
        dd/mm/yyyy, by default True.

        For example July 5th 2015, would be parsed as 07/05/2015 instead of
        05/07/2015.

        Parameters
        ----------
        value : bool
            Whether to parse the date in mm/dd/yyyy format instead of
            dd/mm/yyyy.
        """
        return self._set(readMonthFirst=value)

    def setDefaultDayWhenMissing(self, value):
        """Sets which day to set when it is missing from parsed input,
        by default 1.

        Parameters
        ----------
        value : int
            [description]
        """
        return self._set(defaultDayWhenMissing=value)

    def setAnchorDateYear(self, value):
        """Sets an anchor year for the relative dates such as a day after
        tomorrow. If not set it will use the current year.

        Example: 2021

        Parameters
        ----------
        value : int
            The anchor year for relative dates
        """
        return self._set(anchorDateYear=value)

    def setAnchorDateMonth(self, value):
        """Sets an anchor month for the relative dates such as a day after
        tomorrow. If not set it will use the current month.

        Example: 1 which means January

        Parameters
        ----------
        value : int
            The anchor month for relative dates
        """
        normalizedMonth = value - 1
        return self._set(anchorDateMonth=normalizedMonth)

    def setSourceLanguage(self, value):
        return self._set(sourceLanguage=value)

    def setAnchorDateDay(self, value):
        """Sets an anchor day of the day for the relative dates such as a day
        after tomorrow. If not set it will use the current day.

        Example: 11

        Parameters
        ----------
        value : int
            The anchor day for relative dates
        """
        return self._set(anchorDateDay=value)

    def setRelaxedFactoryStrategy(self, matchStrategy=MatchStrategy.MATCH_FIRST):
        """ Sets matched strategy to search relaxed dates by ordered rules by more exhaustive to less Strategy.

        Not all of the date information needs to be included. For example
        ``"YYYY"`` is also a valid input.

        Parameters
        ----------
        matchStrategy : MatchStrategy
            Matched strategy to search relaxed dates by ordered rules by more exhaustive to less Strategy
        """
        return self._set(relaxedFactoryStrategy=matchStrategy)

    def setAggressiveMatching(self, value):
        """ Sets whether to aggressively attempt to find date matches, even in ambiguous or less common formats

        Parameters
        ----------
        aggressiveMatching : Boolean
            Whether to aggressively attempt to find date matches, even in ambiguous or less common formats
        """
        return self._set(aggressiveMatching=value)


class DateMatcher(AnnotatorModel, DateMatcherUtils):
    """Matches standard date formats into a provided format
    Reads from different forms of date and time expressions and converts them
    to a provided date format.

    Extracts only **one** date per document. Use with sentence detector to find
    matches in each sentence.
    To extract multiple dates from a document, please use the
    :class:`.MultiDateMatcher`.

    Reads the following kind of dates::

        "1978-01-28", "1984/04/02,1/02/1980", "2/28/79",
        "The 31st of April in the year 2008", "Fri, 21 Nov 1997", "Jan 21,
        ‘97", "Sun", "Nov 21", "jan 1st", "next thursday", "last wednesday",
        "today", "tomorrow", "yesterday", "next week", "next month",
        "next year", "day after", "the day before", "0600h", "06:00 hours",
        "6pm", "5:30 a.m.", "at 5", "12:59", "23:59", "1988/11/23 6pm",
        "next week at 7.30", "5 am tomorrow"

    For example ``"The 31st of April in the year 2008"`` will be converted into
    ``2008/04/31``.

    Pretrained pipelines are available for this module, see
    `Pipelines <https://sparknlp.org/docs/en/pipelines>`__.

    For extended examples of usage, see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/MultiDateMatcherMultiLanguage_en.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DATE``
    ====================== ======================

    Parameters
    ----------
    dateFormat
        Desired format for dates extracted, by default yyyy/MM/dd.
    readMonthFirst
        Whether to parse the date in mm/dd/yyyy format instead of dd/mm/yyyy,
        by default True.
    defaultDayWhenMissing
        Which day to set when it is missing from parsed input, by default 1.
    anchorDateYear
        Add an anchor year for the relative dates such as a day after tomorrow.
        If not set it will use the current year. Example: 2021
    anchorDateMonth
        Add an anchor month for the relative dates such as a day after tomorrow.
        If not set it will use the current month. Example: 1 which means January
    anchorDateDay
        Add an anchor day of the day for the relative dates such as a day after
        tomorrow. If not set it will use the current day. Example: 11

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> date = DateMatcher() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("date") \\
    ...     .setAnchorDateYear(2020) \\
    ...     .setAnchorDateMonth(1) \\
    ...     .setAnchorDateDay(11) \\
    ...     .setOutputFormat("yyyy/MM/dd")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     date
    ... ])
    >>> data = spark.createDataFrame([["Fri, 21 Nov 1997"], ["next week at 7.30"], ["see you a day after"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("date").show(truncate=False)
    +-------------------------------------------------+
    |date                                             |
    +-------------------------------------------------+
    |[[date, 5, 15, 1997/11/21, [sentence -> 0], []]] |
    |[[date, 0, 8, 2020/01/18, [sentence -> 0], []]]  |
    |[[date, 10, 18, 2020/01/12, [sentence -> 0], []]]|
    +-------------------------------------------------+

    See Also
    --------
    MultiDateMatcher : for matching multiple dates in a document
    """

    name = "DateMatcher"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DATE

    @keyword_only
    def __init__(self):
        super(DateMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.DateMatcher")
        self._setDefault(
            inputFormats=[""],
            outputFormat="yyyy/MM/dd",
            readMonthFirst=True,
            defaultDayWhenMissing=1,
            anchorDateYear=-1,
            anchorDateMonth=-1,
            anchorDateDay=-1
        )
