package com.johnsnowlabs.nlp.annotators.parser.typdep.util;

public class DependencyLabel {

    private String dependency;
    private String label;
    private int head;
    private int begin;
    private int end;

    public String getDependency() {
        return dependency;
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

    public DependencyLabel(String dependency, String label, int head, int begin, int end) {
        this.dependency = dependency;
        this.label = label;
        this.head = head;
        this.begin = begin;
        this.end = end;
    }
}
