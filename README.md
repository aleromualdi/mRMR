# mRMR
Information based feature selection algorithm based on the mutual information criteria of
max-relevance and min-redundancy presented by Peng *et al.* in http://ieeexplore.ieee.org/document/1453511/. <br/>
<br/>
Because the algorithm makes use of information metric functions, such as Shanon Entropy, it requires discretized data.

# Usage

```
from mrmr import MRMR

selection = MRMR(n_features=5)
indices = selection.fit(X, y)

X_selection = X[:, indices]

```

# General theory

In a dataset with correlated features, using the mutual information between single feature vectors and the target vector as criterion for feature selection may result in selecting a fature set with some redundancy.<br/>
Within the selected features set, features that depend on each other would not improve class-discriminative power and could therefore be removed.

The goal of this algorithm is to find a set of feature vectors $\subseteq X$ with *m* features from the feature matrix $X$, which jointly have the largest dependency on the target class $y$, and low redundancy.<br/>

The computation is performed incrementally, starting from the feature that shows larger mutual information with the *y*, and then selecting the $m^{\text{th}}$ feature from the set $\{X - S_{m-1}\}$

that maximize:

$\hspace{3cm}\max_{x_j \in X - S_{m - 1}} \left[ I(x_i, y) - \frac{1}{m-1} \sum_{x_i \in X - S_{m - 1}} I(x_i, x_j) \right] $


where $=I(x_i, y)$ is the feature-class mutual information ; and $I(x_i, x_j)$ is the feature-feature mutual information.



