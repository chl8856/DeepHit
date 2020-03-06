# DeepHit
Title: "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks"

Authors: Changhee Lee, William R. Zame, Jinsung Yoon, Mihaela van der Schaar

- Reference: C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks," AAAI Conference on Artificial Intelligence (AAAI), 2018
- Paper: http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
- Supplementary: http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit_Appendix

### Description of the code
This code shows the modified implementation of DeepHit on Metabric (single risk) and Synthetic (competing risks) datasets.

The detailed modifications are as follows:
- Hyper-parameter opimization using random search is implemented
- Residual connections are removed
- The definition of the time-dependent C-index is changed; please refer to T.A. Gerds et al, "Estimating a Time-Dependent Concordance Index for Survival Prediction Models with Covariate Dependent Censoring," Stat Med., 2013
- Set "EVAL_TIMES" to a list of evaluation times of interest for optimizating the network with respect these evaluation times.
