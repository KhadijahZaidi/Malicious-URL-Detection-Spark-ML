# Distributed Malicious URL Detection with Spark ML

*Academic group project — Machine Learning*

## Purpose
Build a distributed pipeline that transforms URL and webpage-content data into numerical features and classifies websites as legitimate or malicious.

## Overview
Collaborated on a distributed machine-learning pipeline for classifying websites as legitimate or malicious. PySpark was used to clean and prepare URL and webpage-content data, followed by tokenisation, feature hashing, and TF-IDF transformation. Similar records were explored through clustering before Random Forest, Logistic Regression, and Decision Tree classifiers were trained, tuned, and evaluated. The recorded experiments achieved approximately 99.8% accuracy on the supplied dataset. The project also considered privacy, bias, model reliability, and responsible disclosure. Results are presented as dataset-specific findings rather than guaranteed real-world detection performance.

## Technical Highlights
- Prepared high-dimensional text features from both URL strings and webpage content.
- Compared multiple classifiers and used tuning and evaluation workflows to assess performance.
- Reported limitations carefully because very high test accuracy may not transfer to new or evolving threats.

## Tech Stack
Python, PySpark, Spark ML, TF-IDF, K-means Clustering, Random Forest, Logistic Regression, Decision Tree

## Summary
Developed a distributed malicious-URL detection pipeline with PySpark, TF-IDF, and Spark ML, comparing multiple classification models and achieving approximately 99.8% accuracy on the supplied dataset.

---
Note: this was a group project — my contribution is described above.
