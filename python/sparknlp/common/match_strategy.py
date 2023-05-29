#  Copyright 2017-2023 John Snow Labs
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

"""Allowed strategies for RuleFactory applications regarding replacement"""


class MatchStrategy(object):
    """Object that contains constants for how for matched strategies used in RuleFactory.

    Possible values are:

    ================================== ===============================================================================
    Value                              Description
    ================================== ===============================================================================
    ``MatchStrategy.MATCH_ALL``        This strategy matches all occurrences of all rules in the given text.
    ``MatchStrategy.MATCH_FIRST``      This strategy matches only the first occurrence of each rule in the given text.
    ``MatchStrategy.MATCH_COMPLETE``   This strategy matches only the first occurrence of each rule in the given text.
    ================================== ===============================================================================
    """
    MATCH_ALL = "MATCH_ALL"
    MATCH_FIRST = "MATCH_FIRST"
    MATCH_COMPLETE = "MATCH_COMPLETE"
