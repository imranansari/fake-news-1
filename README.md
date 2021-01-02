# Fake News Application

- TODO
  - Normalize/filter data
  - Write data unit tests with great expectations
    - Include a test for checking record # to prevent duplicate bug
  - Write featurization/pipeline unit tests
    - featurization utils
    - overfit on subset
    - test output shape of model prediction/output characteristics (do values make sense)
    - ensure test score is greater than some threshold

  - Setup DVC pipeline for featurization/running model etc.
  - Setup MLflow for logging of metrics, etc.
  - Setup github action for running tests
  - Incorporate more features from jupyter notebook
  - Error analysis/model interpretation with SHAP etc.
    - http://blog.datadive.net/interpreting-random-forests/
  - BERT/Roberta Baseline
  - Deployment


This holds the full repo for the series of blog posts describing how to build a fake news detection application from ideation to deployment. They are included here:
- [Inital Setup and Tooling](https://www.blog.confetti.ai/post/machine-learning-from-ideation-to-deployment-setting-up)
- [Exploratory Data Analysis](https://www.blog.confetti.ai/post/fake-news-detection-from-ideation-to-deployment-exploratory-data-analysis)
- Building a Model V1 (coming soon)
- Error Analysis and Model V2 (coming soon)
- Model Deployment (coming soon)
