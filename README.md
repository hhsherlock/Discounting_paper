# Discounting Paper

Repository containing code and analyses for the public paper publication.

Two main folders:

## inference

- **different_time_perception**: infer latent parameters using multiple models, with and without context separation. The later sections also include WAIC score calculation based on parameters sampled from the guide.

- **group_by_delay**: rearrange the dataset so that trials with the same delay are grouped together, independent of participant, trial, or context.

- **posterior_samples_bayes_factors**: calculate Bayes factors for inferred parameters across different contexts.

## parameter_recovery

- **parameter_recovery_analysis**: analyze and visualize relationships between sampled and recovered parameters.

- **parameter_recovery_sigmoid**, **parameter_recovery_square_root**, **parameter_recovery_tanh**: parameter recovery pipelines for the three models used in the paper.