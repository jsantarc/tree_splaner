# TreeSplaner

TreeSplaner is a Python library designed to interpret tree-based machine learning models, such as decision trees from `scikit-learn`, by converting their decision rules into natural language. This can be useful for model interpretability, helping users understand the structure and logic behind predictions in a clear and accessible way.

## Features

- Converts decision tree rules to natural language descriptions.
- Provides text-based explanations for predictions on specific samples.
- Outputs impurity information for each decision path, indicating classification certainty.

## How it works 

$\hat{f}(x) = \sum_{m=1}^M c_m \mathbb{I}\{\mathbf{x} \in R_m\}$

- $ R_m $: A region (subset) of the feature space where $ \mathbf{x} $ falls into exactly one leaf node.
- $ c_m $: The predicted value for all data points within region $ R_m $, typically the mode (for classification) of the training samples in $ R_m $.
- $ \mathbb{I}\{\mathbf{x} \in R_m\} $: An indicator function that returns 1 if $ \mathbf{x} $ belongs to region $ R_m $, otherwise 0.

## Interpreting $ \mathbf{x} \in R_m $

The condition $ \mathbf{x} \in R_m $ can be interpreted as a set of **logical AND** conditions that define a region $ R_m $ in the feature space. For example:
- $ R_m = \{x_1 > 1 \text{ AND } x_2 < 3\} $
- Here, $ x_1 > 1 $ and $ x_2 < 3 $ are constraints that an instance $ \mathbf{x} $ must satisfy to belong to the region $ R_m $.

## On the Sum $ \sum_{m=1}^M $

The summation $ \sum_{m=1}^M $ effectively acts like a series of ORs because $ \mathbf{x} $ belongs to exactly one region $ R_m $. However, the regions themselves are mutually exclusive and collectively exhaustive, meaning:

- $ \mathbb{I}\{\mathbf{x} \in R_m\} $ evaluates to 1 for only one $ m $, and 0 for all others.
- As a result, $ \sum_{m=1}^M c_m \mathbb{I}\{\mathbf{x} \in R_m\} $ simplifies to $ c_k $ for the specific region $ R_k $ where $ \mathbf{x} $ lies.

Thus, while the sum combines the contributions from all regions, in practice, it is equivalent to selecting the single $ c_m $ associated with the region $ \mathbf{x} $ belongs to. This resembles the logical OR in terms of selecting one condition.

## Breaking Down the Tree Decision Rule

The decision rule of a tree can be broken down as follows:
- If $ \mathbf{x} \in R_1 $, then predict $ c_1 $.
- If $ \mathbf{x} \in R_2 $, then predict $ c_2 $.
- ...
- Continue for all $ R_m $ regions.
Where each point is an or
This stepwise structure highlights the relationship between the partitioned feature space and the predicted outcome, making the decision tree interpretable and logical.
