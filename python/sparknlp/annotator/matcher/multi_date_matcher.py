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
"""Contains classes for MultiDateMatcher."""

from sparknlp.common import *
from sparknlp.annotator.matcher.date_matcher import DateMatcherUtils


class MultiDateMatcher(AnnotatorModel, DateMatcherUtils):
    """Matches standard date formats into a provided format.

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

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/MultiDateMatcherMultiLanguage_en.ipynb>`__.

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
    >>> date = MultiDateMatcher() \\
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
    >>> data = spark.createDataFrame([["I saw him yesterday and he told me that he will visit us next week"]]) \\
    ...     .toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(date) as dates").show(truncate=False)
    +-----------------------------------------------+
    |dates                                          |
    +-----------------------------------------------+
    |[date, 57, 65, 2020/01/18, [sentence -> 0], []]|
    |[date, 10, 18, 2020/01/10, [sentence -> 0], []]|
    +-----------------------------------------------+
    """

    name = "MultiDateMatcher"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DATE

    @keyword_only
    def __init__(self):
        super(MultiDateMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.MultiDateMatcher")
        self._setDefault(
            inputFormats=[""],
            outputFormat="yyyy/MM/dd",
            readMonthFirst=True,
            defaultDayWhenMissing=1
        )
