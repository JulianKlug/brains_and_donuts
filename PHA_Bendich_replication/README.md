# Replication of PERSISTENT HOMOLOGY ANALYSIS OF BRAIN ARTERY TREES by Paul Bendich et al.

Original Paper: Bendich P, Marron JS, Miller E, Pieloch A, Skwerer S. Persistent homology analysis of brain artery trees. arXiv:14116652 [stat] [Internet]. 2014 Nov 24 [cited 2020 Jan 11]; Available from: http://arxiv.org/abs/1411.6652

## Setting

Replication of the results shown in the above mentioned study on the 2016-2017 Geneva Stroke Dataset.

Input: vascular tree extracted from angioCT from 113 patients with acute ischemic stroke

## Results 
### Age Analysis

##### *PC1 vs. Age*
Distance-Bound on edges | H0 Persistence | H1 Persistence
--- | --- | ---
10 voxels | ![PC1 vs age in H0 Age analysis](./figures/d10/figure4.jpg?ra=true "PC1 vs age in H0 Age analysis") | ![PC1 vs age in H1 Age analysis](./figures/d10/figure6.jpg?ra=true "PC1 vs age in H1 Age analysis")
5 voxels | ![PC1 vs age in H0 Age analysis](./figures/d5/figure4.jpg?ra=true "PC1 vs age in H0 Age analysis") | ![PC1 vs age in H1 Age analysis](./figures/d5/figure6.jpg?ra=true "PC1 vs age in H1 Age analysis")

##### *PC1 vs. PC2*
Distance-Bound on edges | H0 Persistence | H1 Persistence
--- | --- | ---
10 voxels | ![PC1 vs PC2 in H0 Age analysis](./figures/d10/figure3.jpg?ra=true "PC1 vs age in H0 Age analysis") | ![PC1 vs PC2 in H1 Age analysis](./figures/d10/figure5.jpg?ra=true "PC1 vs age in H1 Age analysis")
5 voxels | ![PC1 vs PC2 in H0 Age analysis](./figures/d5/figure3.jpg?ra=true "PC1 vs age in H0 Age analysis") | ![PC1 vs PC2 in H1 Age analysis](./figures/d5/figure5.jpg?ra=true "PC1 vs age in H1 Age analysis")


### Gender Analysis

Distance-Bound on edges | H0 Persistence | H1 Persistence
--- | --- | ---
10 voxels | ![Gender difference in H0 Persistence](./figures/d10/figure7.jpg?ra=true "Gender difference in H0 Persistence") | ![Gender difference in H1 Persistence](./figures/d10/figure8.jpg?ra=true "Gender difference in H1 Persistence")
5 voxels | ![Gender difference in H0 Persistence](./figures/d5/figure7.jpg?ra=true "Gender difference in H0 Persistence") | ![Gender difference in H1 Persistence](./figures/d5/figure8.jpg?ra=true "Gender difference in H1 Persistence")

 
### PCA 

Distance-Bound on edges | H0 Persistence | H1 Persistence
--- | --- | ---
10 voxels | ![H0 PCA Representation](./figures/d10/figure1.jpg?ra=true "H0 PCA Representation") | ![H1 PCA Representation](./figures/d10/figure2.jpg?ra=true "H0 PCA Representation1")
5 voxels | ![H0 PCA Representation](./figures/d5/figure1.jpg?ra=true "H0 PCA Representation") | ![H1 PCA Representation](./figures/d5/figure2.jpg?ra=true "H0 PCA Representation1")


## Further Analysis
### Stroke volume analysis

##### *PC1 vs. Stroke Volume*
Distance-Bound on edges | H0 Persistence | H1 Persistence
--- | --- | ---
10 voxels | ![PC1 vs stroke volume in H0 Persistence](./figures/d10/H0_volume.jpg?ra=true "PC1 vs stroke volume in H0 Persistence") | ![PC1 vs stroke volume in H1 Persistence](./figures/d10/H0_volume.jpg?ra=true "PC1 vs stroke volume in H1 Persistence")
5 voxels | ![PC1 vs stroke volume in H0 Persistence](./figures/d5/H0_volume.jpg?ra=true "PC1 vs stroke volume in H0 Persistence") | ![PC1 vs stroke volume in H1 Persistence](./figures/d5/H0_volume.jpg?ra=true "PC1 vs stroke volume in H1 Persistence")


##### *PC1 vs. PC2*
Distance-Bound on edges | H0 Persistence | H1 Persistence
--- | --- | ---
10 voxels | ![PC1 vs PC2 in H0 Persistence](./figures/d10/H0_volume_intercorr.jpg?ra=true "PC1 vs PC2 in H0 Persistence") | ![PC1 vs PC2 in H1 Persistence](./figures/d10/H1_volume_intercorr.jpg?ra=true "PC1 vs PC2 in H1 Persistence")
5 voxels | ![PC1 vs PC2 in H0 Persistence](./figures/d5/H0_volume_intercorr.jpg?ra=true "PC1 vs PC2 in H0 Persistence") | ![PC1 vs PC2 in H1 Persistence](./figures/d5/H1_volume_intercorr.jpg?ra=true "PC1 vs PC2 in H1 Persistence")




 