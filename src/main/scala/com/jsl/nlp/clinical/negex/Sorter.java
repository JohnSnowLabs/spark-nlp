package com.jsl.nlp.clinical.negex;

import java.util.*;
import java.lang.*;
import java.io.*;

// Utility class to sort the negation rules by length in descending order.
// Rules need to be matched by longest first because there is overlap between the
// RegEx of the rules.
//

// Author: Imre Solti
// solti@u.washington.edu
// Date: 10/20/2008


public class Sorter {

    public ArrayList sortRules(ArrayList unsortedRules) {

        try {
            // Sort the negation rules by length to make sure
            // that longest rules match first.
            String temp = "";
            for (int i = 0; i<unsortedRules.size()-1; i++) {
                for (int j = i+1; j<unsortedRules.size(); j++) {
                    String a = (String) unsortedRules.get(i);
                    String b = (String) unsortedRules.get(j);
                    if (a.trim().length()<b.trim().length()) {
                        // Sorting into descending order by lebgth of string.
                        unsortedRules.set(i, b);
                        unsortedRules.set(j, a);
                    }
                }
            }
        }
        catch (Exception e) {
            System.out.println(e);
        }
        return unsortedRules;
    }
}
