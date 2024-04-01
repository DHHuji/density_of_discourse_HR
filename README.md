# density_of_discourse_HR

This repository contains the replication code for the paper "The Discursive Evolution of Human Rights Law: Empirical Insights from a Computational Analysis of 180,000 UN Recommendations."

## Files:

1. `LDACoherenceTest.py`: This code is used to find the optimal number of topics for the Latent Dirichlet Allocation (LDA) topic modeling algorithm.

2. `LDA.py`: The main code implementing the LDA algorithm for topic modeling.

3. `byyearbymech.py`: This script counts the documents per year per UN mechanism (third).

4. `WCountries`: This file is needed for preprocessing.

5. `Countries and nationalities`: Another file required for preprocessing.

## Usage:
- Run `LDACoherenceTest.py` to determine the optimal number of topics.
- Then, use `LDA.py` to perform topic modeling based on the determined number of topics.
- Finally, utilize `byyearbymech.py` to count the documents per year per UN mechanism.

Please ensure all necessary files are present and configured properly before running the scripts.
