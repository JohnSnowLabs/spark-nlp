---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Productivity
permalink: /docs/en/alab/productivity
key: docs-training
modify_date: "2022-12-11"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
  nav: annotation-lab
---

<style>
bl {
  font-weight: 400;
}

es {
  font-weight: 400;
  font-style: italic;
}
</style>

## Analytics Charts

By default, the Analytics page is disabled for every project because computing the analytical charts is a resource-intensive task and might temporarily influence the responsiveness of the application, especially when triggered in parallel with other training/preannotation jobs. However, users can file a request to enable the Analytics page which can be approved by any [admin user](docs/en/alab/user_management#user-groups). The request is published on the [Analytics Requests](/docs/en/alab/analytics_permission) page, visible to any <es>admin</es> user. Once the <es>admin</es> user approves the request, any team member can access the Analytics page.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/enable_analytics.gif" style="width:100%;"/>

A refresh button is present on the top-right corner of the Analytics page. The Analytics charts doesn't automatically reflect the changes made by the annotators (like creating tasks, adding new completion, etc.). Updating the analytics to reflect the latest changes can be done using the refresh button.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/refresh.png" style="width:100%;"/>

### Task Analytics

To access Task Analytics, navigate on the first tab of the <es>Analytics</es> Dashboard, called <bl>Tasks</bl>. The following blog post explains how to [Improve Annotation Quality using Task Analytics in the Annotation Lab](https://www.johnsnowlabs.com/improving-annotation-quality-using-analytics-in-the-annotation-lab/).

Below are the charts included in the Tasks section.

**Total number of task in the Project**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/total_tasks.png" style="width:100%;"/>

**Total number of task in a Project in last 30 days**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/total_tasks_last_30_days.png" style="width:100%;"/>

**Breakdown of task in the Project by Status**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/tasks_by_status.png" style="width:100%;"/>

**Breakdown of task by author**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/tasks_created_by.png" style="width:100%;"/>

**Summary of task status for each annotator**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/annotator_summary.png" style="width:100%;"/>

**Total number of label occurrences across all completions**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/total_label_frequency_across_completions.png" style="width:100%;"/>

**Average number of label occurrences for each completion**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/average_label_frequency_per_completion.png" style="width:100%;"/>

**Total number of label occurrences across all completions for each annotator**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/label_frequency_by_annotator.png" style="width:100%;"/>

**Total vs distinct count of labels across all completions**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/total_vs_distinct_by_label_across_completions.png" style="width:100%;"/>

**Average number of tokens by label**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/average_number_of_tokens_by_label.png" style="width:100%;"/>

**Total number of label occurrences that include numeric values**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/numeric_values_across_labels.png" style="width:100%;"/>

<br />

### Team Productivity

To access Team Productivity charts, navigate on the second tab of the <es>Analytics</es> Dashboard, called <bl>Team Productivity</bl>. The following blog post explains how to [Keep Track of Your Team Productivity in the Annotation Lab](https://www.johnsnowlabs.com/keep-track-of-your-team-productivity-in-the-annotation-lab/).

Below are the charts included in the Team Productivity section.

**Total number of completions in the Project**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/total_completions.png" style="width:100%;"/>

**Total number of completions in the Project in the last 30 days**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/total_completions_in_last_30_days.png" style="width:100%;"/>

**Total number of completions for each Annotator**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/total_number_of_completions_per_annotator.png" style="width:100%;"/>

**Total number of completions submitted over time for each Annotator**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/submitted_completions_over_time_per_annotator.png" style="width:100%;"/>

**Average time spent by the Annotator in each task**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/average_time_annotator_spent_on_one_task.png" style="width:100%;"/>

**Total number of completions submitted over time**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/submitted_completions_over_time.png" style="width:100%;"/>

<br />

### Inter-Annotator Agreement (IAA)

Starting from version 2.8.0, Inter Annotator Agreement(IAA) charts allow the comparison between annotations produced by <es>Annotators</es>, <es>Reviewers</es>, or <es>Managers</es>.

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/iaa_analytics.gif" style="width:100%;"/>

Inter Annotator Agreement charts can be used by <es>Annotators</es>, <es>Reviewers</es>, and <es>Managers</es> for identifying contradictions or disagreements within the starred completions (Ground Truth). When multiple annotators work on same tasks, IAA charts are handy to measure how well the annotations created by different annotators align. IAA chart can also be used to identify outliers in the labeled data, or to compare manual annotations with model predictions.

To access IAA charts, navigate on the third tab of the <es>Analytics</es> Dashboard of NER projects, called <bl>Inter-Annotator Agreement</bl>. Several charts should appear on the screen with a default selection of annotators to compare. The dropdown selections on top-left corner of each chart allow you to change annotators for comparison purposes. There is another dropdown to select the label type for filtering between NER labels and Assertion Status labels for projects containing both NER and Assertion Status entities. It is also possible to download the data generated for some chart in CSV format by clicking the download button just below the dropdown selectors.

> **Note:** Only the <es>Submitted</es> and <es>starred (Ground Truth)</es> completions are used to render these charts.

The following blog post explains how your team can [Reach Consensus Faster by Using IAA Charts in the Annotation Lab](https://www.johnsnowlabs.com/reach-consensus-faster-by-using-iaa-charts-in-the-annotation-lab/).

Below are the charts included in the Inter-Annotator Agreement section.

**High-level IAA between annotators on all common tasks**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/high_level_iaa_between_annotators_on_all_common_tasks.png" style="width:100%;"/>

**IAA between annotators for each label on all common tasks**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/iaa_between_annotators_on_all_common_tasks.png" style="width:100%;"/>

**Comparison of annotations by annotator on each chunk**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/annotators_comparison_by_chunk.png" style="width:100%;"/>

**Comparison of annotations by model and annotator (Ground Truth) on each chunk**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/prediction_vs_groundtruth.png" style="width:100%;"/>

**All chunks annotated by an annotator**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/all_chunks_extracted_by_annotator.png" style="width:100%;"/>

**Frequency of labels on chunks annotated by an annotator**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/chunks_extracted_by_annotator.png" style="width:100%;"/>

**Frequency of a label on chunks annotated by each annotator**

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/analytics/chunks_extracted_by_label.png" style="width:100%;"/>

## Download data used for charts

CSV file for specific charts can be downloaded using the new download button which will call specific API endpoints: /api/projects/{project_name}/charts/{chart_type}/download_csv

![Screen Recording 2022-03-08 at 3 47 49 PM](https://user-images.githubusercontent.com/17021686/158564836-691a2b79-f3ca-4317-ad31-51cfbc9d71df.gif)
