package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

public class DependencyLabel {

    private String token;
    private String label;
    private int head;
    private int begin;
    private int end;

    public String getToken() {
        return token;
    }

    public String getLabel() {
        return label;
    }

    public int getHead() {
        return head;
    }

    public int getBegin() {
        return begin;
    }

    public int getEnd() {
        return end;
    }

    public DependencyLabel(String token, String label, int head, int begin, int end) {
        this.token = token;
        this.label = label;
        this.head = head;
        this.begin = begin;
        this.end = end;
    }
}
