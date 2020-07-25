# Project journal
## Implementing basic pipelines 

|Start Date|End Date  |
|----------|----------|
|2020-04-01|2020-05-30|

### Description

Implemented basic pipelines for topological and non-topological stroke prediction based on receptive fields. 

**Example Pipeline overview**
1. Subimage creatio (Receptive fields) with multiple widths via `RollingSubImageTransformer`
2. Feature creation: Compute persistance in homology dimensions (0, 1, 2) and obtain persistence entropies and wasserstein amplitude per subimage for subimages of different widths
3. Split Train/Test data
4. Fit and test prediction of voxel-wise infarction with `RandomForestClassifier`


### Delivrables

- [x] topological feature creation and analysis pipeline based on giotto
- [x] receptive field pipeline as a baseline for comparison
- [x] working example [notebook](./data_exploration/topological_feature_creation_and_classification_pipeline.ipynb) 

### Results 

The basic implementation of this pipeline does however not seem to learn anything yet, and feature creation remains very slow:
```
Dice: 0.0
ROC AUC: 0.49997742739266
Feature Creation timing: 46438.243881464005
Feature Classification and Prediction timing: 5029.564110517502
```
## Normalisation/Standardisation 

|Start Date|End Date  |
|----------|----------|
|2020-06-01|2020-06-XX|

### Description

Data should come from the same distribution. To get back this original distribution, a capping is applied to each sample.

**Capping strategy**
- data is capped to 0-500
- data outside this range does not seem relevant for vascular analysis
- x < 0 -> mostly air
- x > 500 -> mostly rest of bones, some vascular artefacts, calcifications 

### Delivrables

- [x] Visualisation of the normalisation in a [notebook](./data_exploration/normalisation.ipynb).
- [ ] Todo: evaluate relevance of standardisation

|Original dataset| Capped dataset |
|----------|----------|
|![original](./static/journal/original_dataset_histograms.png "Original") | ![capped](./static/journal/capped_dataset_histograms.png "Capped")|

### Conclusion

- Data should be capped between 0-500 Hu

## Data balancing 

|Start Date|End Date  |
|----------|----------|
|2020-06-01|2020-06-20|

### Description

Stroke imaging data is extremely unbalanced with a positive to negative voxel sample ratio of 2:100 (number of infarcted voxels: number of non-infarcted voxels), making data balancing necessary for effective learning algorithms. Two main methods are used here:
- Restriction of training and evaluation to voxels within the brain
- Balance training data for better learning 

### Delivrables

- [x] Mask restriction: restrict training and testing to brain mask 
- [x] Undersampling: rebalance training data by undersampling negative samples (non infarcted) to obtain a 1:1 ratio

|Working example| Results |
|----------|----------|
|![Working example](./static/journal/masked_undersampled_working_example.png "Left: Output probability / Right: GT") | Train Dice: 1.0 <br>Train ROC AUC: 1.0 <br>Test Dice: 0.04665507672762529 <br>Test ROC AUC: 0.6557306214976681 <br>Feature Creation timing: 676.7280170917511 <br>Feature Classification and Prediction timing: 111.9262273311615|

### Conclusion

- Balancing is necessary for a learning algorithm for this dataset.
- Balancing speeds up feature creation significantly.

## Hemispheric stroke presence classification

|Start Date|End Date  |
|----------|----------|
|2020-07-01|2020-07-25|

### Description

Prediction of stroke segmentation is a very difficult task. We have therefore tried to reduce the complexity of this problem to find relevant parameters by creating a hemispheric stroke presence classification task.

To this end, all volumes in the dataset were split into two hemispheres, and models were evaluated on their ability to predict the presence of an infarct on angioCT imaging in the whole hemisphere. 

A grid search was done with a 3-way train, validation, test split and relevant models were identified from their validation score, to then be evaluated on the test set. 

``` python
params_search_space = {
    'homology_dimensions': [[0, 1, 2]],
    'features': [['PersistenceEntropy', 'Amplitude','NumberOfPoints']],
    'amplitude_metric': ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat', 'silhouette', 'persistence_image', 'landscape'],
    'processing_filter': [True, False],
    'processing_scaler_metric': ['None', 'bottleneck', 'wasserstein', 'landscape', 'betti', 'heat', 'silhouette', 'persistence_image', 'landscape'],
    'inverse_input': [True, False],
    'subsampling_factor': [1, 2, 3],
    'model': ['RandomForestClassifier', 'LogisticRegression']
}
```

All features on all homology dimensions were always used for the grid search to reduce search space, as all used models could easily reduce their weights to 0. 

Unfortunately, none of the identified models achieved reasonable results on the test set. 

### Delivrables

- [x] Hemispheric dataset
- [x] Grid search
- [x] Data inversion function (f(x): max(X) - X)
- [x] Grid search results: [LogisticRegression Model](./static/results/hemisphere_classification/lreg_grid_search.csv), [RandomForestClassifier Model](./static/results/hemisphere_classification/tree_grid_search.csv) 

### Models with best validation results

|LogisticRegression Model| RandomForestClassifier Model |
|----------|----------|
| Accuracies:  <br> train: 0.65, val: 0.63, test: 0.54 | Accuracies:  <br> train: 1, val: 0.61, test: 0.56|

### Conclusion

- Generating topologic features from the whole hemisphere on angioCT did not yield good results. 
- Topological features should be searched on a smaller scale (ie. receptive fields)

