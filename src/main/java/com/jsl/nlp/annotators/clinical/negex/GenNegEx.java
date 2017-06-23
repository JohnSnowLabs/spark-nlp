package com.jsl.nlp.annotators.clinical.negex;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.*;


/***************************************************************************************
 * Author: Imre Solti
 * Date: 09/15/2008
 * Modified: 04/15/2009
 *	Changed to specifications of test kit and discussions with WC and PH.
 * Modified: 04/26/2009
 * Fixed the deletion of last character in scope fo PREN, PREP negation scopes.
 *
 * Wendy Chapman's NegEx algorithm in Java.
 *
 * Sentence boundaries serve as WINDOW for negation (suggested by Wendy Chapman)
 *
 ****************************************************************************************/

/*
Copyright 2008 Imre Solti

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
limitations under the License.
*/

public class GenNegEx {

    private static final boolean NEGATE_POSSIBLE = true;
    private static final List<String> rules = new ArrayList<>();

    static {
        Scanner ruleScanner = new Scanner(GenNegEx.class.getClass().getResourceAsStream("/negex.rules.txt"));
        while (ruleScanner.hasNextLine()) {
            rules.add(ruleScanner.nextLine());
        }
        ruleScanner.close();
        // There is efficiency issue here. It is better if rules are sorted by the
        // calling program once and used without sorting in GennegEx.
        Sorter.sortRules(rules);
    }

    public static String negCheck(String sentenceString) {

        String filler = "_";

        // Sort the rules by length in descending order.
        // Rules need to be sorted so the longest rule is always tried to match
        // first.
        // Some of the rules overlap so without sorting first shorter rules (some of them POSSIBLE or PSEUDO)
        // would match before longer legitimate negation rules.
        //

        // Process the sentence and tag each matched negation
        // rule with correct negation rule tag.
        // In both the negation rules and in the  phrase replace white space
        // with "filler" string. (This could cause problems if the sentences
        // we study has "filler" on their own.)

        // Sentence needs one character in the beginning and end to match.
        // We remove the extra characters after processing.
        String sentence = "." + sentenceString + ".";

        for (String rule : rules) {
            Pattern p = Pattern.compile("[\\t]+");    // Working.
            String[] ruleTokens = p.split(rule.trim());
            // Add the regular expression characters to tokens and asemble the rule again.
            String[] ruleMembers = ruleTokens[0].trim().split(" ");
            String rule2 = "";
            for (int i = 0; i <= ruleMembers.length - 1; i++) {
                if (!ruleMembers[i].equals("")) {
                    if (ruleMembers.length == 1) {
                        rule2 = ruleMembers[i];
                    } else {
                        rule2 += ruleMembers[i].trim() + "\\s+";
                    }
                }
            }
            // Remove the last s+
            if (rule2.endsWith("\\s+")) {
                rule2 = rule2.substring(0, rule2.lastIndexOf("\\s+"));
            }

            rule2 = "(?m)(?i)[[\\p{Punct}&&[^\\]\\[]]|\\s+](" + rule2 + ")[[\\p{Punct}&&[^_]]|\\s+]";

            Pattern p2 = Pattern.compile(rule2.trim());
            Matcher m = p2.matcher(sentence);

            while (m.find()) {
                sentence = m.replaceAll(" " + ruleTokens[1].trim()
                        + m.group().trim().replaceAll(" ", filler)
                        + ruleTokens[1].trim() + " ");
            }
        }

        // Because PRENEGATION [PREN} is checked first it takes precedent over
        // POSTNEGATION [POST].
        // Similarly POSTNEGATION [POST] takes precedent over POSSIBLE PRENEGATION [PREP]
        // and [PREP] takes precedent over POSSIBLE POSTNEGATION [POSP].

        Pattern pSpace = Pattern.compile("[\\s+]");
        String[] sentenceTokens = pSpace.split(sentence);
        StringBuilder sb = new StringBuilder();


        // Check for [PREN]
        for (int i = 0; i<sentenceTokens.length; i++) {
            sb.append(" ").append(sentenceTokens[i].trim());
            if (sentenceTokens[i].trim().startsWith("[PREN]")) {

                for (int j = i+1; j<sentenceTokens.length; j++) {
                    if (sentenceTokens[j].trim().startsWith("[CONJ]") ||
                            sentenceTokens[j].trim().startsWith("[PSEU]") ||
                            sentenceTokens[j].trim().startsWith("[POST]") ||
                            sentenceTokens[j].trim().startsWith("[PREP]") ||
                            sentenceTokens[j].trim().startsWith("[POSP]") ) {
                        break;
                    }
                }
            }
        }

        sentence = sb.toString();
        pSpace = Pattern.compile("[\\s+]");
        sentenceTokens = pSpace.split(sentence);
        StringBuilder sb2 = new StringBuilder();

        // Check for [POST]
        for (int i = sentenceTokens.length-1; i>0; i--) {
            sb2.insert(0, sentenceTokens[i] + " ");
            if (sentenceTokens[i].trim().startsWith("[POST]")) {
                for (int j = i-1; j>0; j--) {
                    if (sentenceTokens[j].trim().startsWith("[CONJ]") ||
                            sentenceTokens[j].trim().startsWith("[PSEU]") ||
                            sentenceTokens[j].trim().startsWith("[PREN]") ||
                            sentenceTokens[j].trim().startsWith("[PREP]") ||
                            sentenceTokens[j].trim().startsWith("[POSP]") ) {
                        break;
                    }
                }
            }
        }

        sentence = sb2.toString();

        // If POSSIBLE negation is detected as negation.
        // negatePossible being set to "true" then check for [PREP] and [POSP].
        if (NEGATE_POSSIBLE) {
            pSpace = Pattern.compile("[\\s+]");
            sentenceTokens = pSpace.split(sentence);

            StringBuilder sb3 = new StringBuilder();

            // Check for [PREP]
            for (int i = 0; i<sentenceTokens.length; i++) {
                sb3.append(" ").append(sentenceTokens[i].trim());
                if (sentenceTokens[i].trim().startsWith("[PREP]")) {

                    for (int j = i+1; j<sentenceTokens.length; j++) {
                        if (sentenceTokens[j].trim().startsWith("[CONJ]") ||
                                sentenceTokens[j].trim().startsWith("[PSEU]") ||
                                sentenceTokens[j].trim().startsWith("[POST]") ||
                                sentenceTokens[j].trim().startsWith("[PREN]") ||
                                sentenceTokens[j].trim().startsWith("[POSP]") ) {
                            break;
                        }
                    }
                }
            }

            sentence = sb3.toString();
            pSpace = Pattern.compile("[\\s+]");
            sentenceTokens = pSpace.split(sentence);
            StringBuilder sb4 = new StringBuilder();

            // Check for [POSP]
            for (int i = sentenceTokens.length-1; i>0; i--) {
                sb4.insert(0, sentenceTokens[i] + " ");
                if (sentenceTokens[i].trim().startsWith("[POSP]")) {
                    for (int j = i-1; j>0; j--) {
                        if (sentenceTokens[j].trim().startsWith("[CONJ]") ||
                                sentenceTokens[j].trim().startsWith("[PSEU]") ||
                                sentenceTokens[j].trim().startsWith("[PREN]") ||
                                sentenceTokens[j].trim().startsWith("[PREP]") ||
                                sentenceTokens[j].trim().startsWith("[POST]") ) {
                            break;
                        }
                    }
                }
            }

            sentence = sb4.toString();
        }

        // Remove the filler character we used.
        sentence = sentence.replaceAll(filler, " ");

        // Remove the extra periods at the beginning
        // and end of the sentence.
        sentence = sentence.substring(0, sentence.trim().lastIndexOf('.'));
        sentence = sentence.replaceFirst(".", "");

        return sentence;
    }
}
