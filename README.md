# TFDBA_Modeling
High-precision data-driven modeling of DAB based on transfer learning by integrating simulated and real-world data.

The setup of the environment and specific operational details of the model will be made public after the paper is accepted.

data_combine_test.py
Comparison experiments on the impact of different data usage schemes on final model accuracy. A model named xgboost1 was directly established using simulated data, while another model named xgboost3 was built using experimental data. Additionally, a model named xgboost4 was created using the accumulation of experimental and simulated data. A linear combination of the model built with simulated data (xgboost1) and the model built with real data (xgboost2) was also implemented.

The performance of the models obtained from these strategies shows a significant gap compared to the TFDBA model.
