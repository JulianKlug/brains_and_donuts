import sys, os, time, pickle

path_bnd = '../../'
sys.path.insert(1, path_bnd)
import analysis_tools.data_loader as dl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gtda.homology import CubicalPersistence
from pgtda.diagrams import PersistenceEntropy, Amplitude, Filtering, Scaler, NumberOfPoints
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from analysis_tools.metrics.plot_ROC import plot_roc
from analysis_tools.metrics.metrics import dice, roc_auc
from analysis_tools.utils.utils import get_undersample_selector_array
from visual_tools.dataset_visualization import visualize_dataset
from analysis_tools.utils.masked_rolling_subimage_transformer import MaskedRollingSubImageTransformer

## Import data
data_dir = '/media/miplab-nas2/Data/klug/geneva_stroke_dataset/working_data/withAngio_all_2016_2017'
save_dir = '/home/klug/output/bnd/feature_eval'
data_set_name = 'data_set.npz'
experiment_name = 'masked_undersampled_data'


n_subjects = 3
n_threads = 50
subsampling_factor = 2
batch_size = 10
compare_with_rf = True

# Create necessary directories to save data
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
experiment_save_dir = os.path.join(save_dir, experiment_name)
if not os.path.exists(experiment_save_dir):
    os.mkdir(experiment_save_dir)
else:
    os.mkdir(f'{experiment_save_dir}_{time.strftime("%Y%m%d_%H%M%S")}')
pickle_dir = os.path.join(experiment_save_dir, 'pickled_data')
if not os.path.exists(pickle_dir):
    os.mkdir(pickle_dir)

# Load data
clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params = \
    dl.load_structured_data(data_dir, data_set_name)

# Reshape ct_inputs as it has 1 channel
ct_inputs = ct_inputs.reshape((*ct_inputs.shape[:-1]))

# Apply brain masks
X = (ct_inputs[:n_subjects] * brain_masks[:n_subjects])[range(n_subjects), ::subsampling_factor, ::subsampling_factor, ::subsampling_factor]
y = (ct_lesion_GT[:n_subjects] * brain_masks[:n_subjects])[range(n_subjects), ::subsampling_factor, ::subsampling_factor, ::subsampling_factor]
mask = brain_masks[:n_subjects][range(n_subjects), ::subsampling_factor, ::subsampling_factor, ::subsampling_factor]
_, n_x, n_y, n_z = mask.shape

# Normalise data
# Capping (threshold to 0-500 as values outside this range seem non relevant to the vascular analysis)
vmin = 0
vmax = 500
X[X < vmin] = vmin
X[X > vmax] = vmax

# Todo to reevaluate relevance of std normalisation

## Feature Creation
width_list = [[3, 3, 3]]
n_widths = len(width_list)
n_widths_for_thread_attribution = 1 # TODO verify sklearn jobs
start = time.time()
# Note that padding should be same so that output images always have the same size
feature_transformers = [
                            PersistenceEntropy(n_jobs=1),
                            Amplitude(metric='wasserstein', metric_params={'p':2}, order=None, n_jobs=1),
                            NumberOfPoints(n_jobs=1)
                        ]
n_subimage_features = len(feature_transformers)
transformer = make_pipeline(
                CubicalPersistence(homology_dimensions=(0, 1 ,2), n_jobs=int(n_threads/n_widths_for_thread_attribution)),
                Filtering(epsilon=np.max(X)-1, below=False),
                Scaler(),
                make_union(*feature_transformers, n_jobs=int((n_threads/n_widths_for_thread_attribution)/n_subimage_features))
              )

# Batch decomposition to spare memory
X_features = []
masked_y = []
masked_subimages = []
for batch_offset in tqdm(range(0, X.shape[0], batch_size)):
    X_batch = X[batch_offset:batch_offset+batch_size]
    mask_batch = mask[batch_offset:batch_offset+batch_size]
    # apply mask on ground truth
    batch_y = y[batch_offset:batch_offset + batch_size]
    masked_batch_y = [batch_y[subj_idx][mask_batch[subj_idx]] for subj_idx in range(mask_batch.shape[0])]
    masked_y = masked_y + masked_batch_y

    masked_subimage_transformer = MaskedRollingSubImageTransformer(mask=mask_batch, width_list=width_list, padding='same')
    batch_masked_subimages = masked_subimage_transformer.fit_transform(X_batch)

    for subj_index in range(X_batch.shape[0]):
        subj_masked_subimages = batch_masked_subimages[subj_index]
        # For each subimage for each subimage width, the features of interest are extracted.
        subj_masked_subfeatures_per_width = [transformer.fit_transform(subj_masked_subimages[width_index]) for
                                             width_index in range(n_widths)]
        # Features are then concatenated along a subimage-width agnostic feature dimension
        subj_masked_subfeatures = np.concatenate(subj_masked_subfeatures_per_width, axis=-1)
        X_features.append(subj_masked_subfeatures)
    if compare_with_rf:
        # flatten along subimage dimensions to obtain (n_samples, n_widths, n_voxels, n_subimage_voxels)
        batch_flat_masked_subimages = [[batch_masked_subimages[subj_idx][width_idx]
                                            .reshape(batch_masked_subimages[subj_idx][width_idx].shape[0], -1)
                                        for width_idx in range(n_widths)] for subj_idx in range(len(X_batch))]
        # join along image widths to obtain (n_samples, n_voxels, n_subimage_voxels_for_all_widths)
        batch_flat_masked_subimages = [np.concatenate(batch_flat_masked_subimages[subj_idx], axis=-1) for subj_idx in
                                       range(len(X_batch))]
        masked_subimages = masked_subimages + batch_flat_masked_subimages

n_features = X_features[0][0].shape[-1]

end = time.time()
feature_creation_timing = end - start
print(f'Features ready after {feature_creation_timing} s')

## Feature Classification
#### Create classifier
start = time.time()
classifier = RandomForestClassifier(n_estimators=10000, n_jobs=-1)

#### Prepare dataset
X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(X_features, masked_y, mask, test_size=0.3, random_state=42)
n_train, n_test = len(X_train), len(X_test)
X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)

# Undersample training data
balancing_selector = get_undersample_selector_array(y_train)
X_train, y_train = X_train[balancing_selector], y_train[balancing_selector]

## save data
pickle.dump(X_train, open(os.path.join(pickle_dir, 'X_train.p'), 'wb'))
pickle.dump(X_test, open(os.path.join(pickle_dir, 'X_test.p'), 'wb'))
pickle.dump(y_train, open(os.path.join(pickle_dir, 'y_train.p'), 'wb'))
pickle.dump(y_test, open(os.path.join(pickle_dir, 'y_test.p'), 'wb'))

#### Train classifier
classifier.fit(X_train, y_train)

### save classifier
pickle.dump(classifier, open(os.path.join(pickle_dir, 'trained_classifier.p'), 'wb'))

#### Apply classifier
test_probas = classifier.predict_proba(X_test)
test_predicted = classifier.predict(X_test)

train_probas = classifier.predict_proba(X_train)
train_predicted = classifier.predict(X_train)

### save predicted output
pickle.dump(test_probas, open(os.path.join(pickle_dir, 'test_probas.p'), 'wb'))
pickle.dump(test_predicted, open(os.path.join(pickle_dir, 'test_predicted.p'), 'wb'))

pickle.dump(train_probas, open(os.path.join(pickle_dir, 'train_probas.p'), 'wb'))
pickle.dump(train_predicted, open(os.path.join(pickle_dir, 'train_predicted.p'), 'wb'))

#### Reconstruct output
end = time.time()
feature_classification_and_prediction_time = end - start

## Model (Features + Classifier) Evaluation
test_dice_score = dice(test_predicted.flatten(), y_test.flatten())
test_roc_auc_score, test_roc_curve_details = roc_auc(y_test, test_predicted)
train_dice_score = dice(train_predicted.flatten(), y_train.flatten())
train_roc_auc_score, train_roc_curve_details = roc_auc(y_train, train_predicted)

print('Train Dice:', train_dice_score)
print('Train ROC AUC:', train_roc_auc_score)
print('Test Dice:', test_dice_score)
print('Test ROC AUC:', test_roc_auc_score)

with open(os.path.join(experiment_save_dir, 'logs.txt'), "a") as log_file:
    log_file.write('Train Dice: %s\n' % train_dice_score)
    log_file.write('Train ROC AUC: %s\n' % train_roc_auc_score)
    log_file.write('Test Dice: %s\n' % test_dice_score)
    log_file.write('Test ROC AUC: %s\n' % test_roc_auc_score)
    log_file.write('Feature Creation timing: %s\n' % feature_creation_timing)
    log_file.write('Feature Classification and Prediction timing: %s\n' % feature_classification_and_prediction_time)

# Get scores by a pure voxel-value based classifier for comparison
if compare_with_rf:
    rf_classifier = clone(classifier)
    masked_subimages_train, masked_subimages_test = train_test_split(masked_subimages, test_size=0.3, random_state=42)
    masked_subimages_train, masked_subimages_test = np.concatenate(masked_subimages_train)[balancing_selector], np.concatenate(masked_subimages_test)
    rf_classifier.fit(masked_subimages_train, y_train)
    rf_test_predicted = rf_classifier.predict(masked_subimages_test)
    rf_test_dice_score = dice(rf_test_predicted.flatten(), y_test.flatten())
    rf_test_roc_auc_score, _ = roc_auc(y_test, rf_test_predicted)
    print('Rf-only test Dice:', rf_test_dice_score)
    print('Rf-only test ROC AUC:', rf_test_roc_auc_score)
    with open(os.path.join(experiment_save_dir, 'logs.txt'), "a") as log_file:
        log_file.write('Rf-Train Dice: %s\n' % rf_test_dice_score)
        log_file.write('Rf-Train ROC AUC: %s\n' % rf_test_roc_auc_score)

# %%
test_fpr, test_tpr, roc_thresholds = test_roc_curve_details
plot_roc([test_tpr], [test_fpr], save_dir=experiment_save_dir, save_plot=True, model_name='test_' + experiment_name)
train_fpr, train_tpr, roc_thresholds = train_roc_curve_details
plot_roc([train_tpr], [train_fpr], save_dir=experiment_save_dir, save_plot=True, model_name='train_' + experiment_name)

## Model feature analysis
#### Model confusion matrix
confusion = confusion_matrix(y_test, test_predicted)
plt.imshow(confusion)
plt.savefig(os.path.join(experiment_save_dir, experiment_name + '_confusion_matrix.png'))

#### Feature correalation
correlation = np.abs(np.corrcoef(X_train.T))
plt.imshow(correlation)
plt.savefig(os.path.join(experiment_save_dir, experiment_name + '_correlation_matrix.png'))

#### Reconstruct output and label
vxl_index = 0
background_value = 0
test_3D_probas = np.full((n_test, n_x, n_y, n_z, test_probas.shape[-1]), background_value, dtype=np.float64) # fill shape with background
test_3D_y = np.full(mask_test.shape, 0, dtype=np.int)
for subj_idx in range(n_test):
    subj_n_vxl = np.sum(mask_test[subj_idx])
    subj_3d_probas = test_probas[vxl_index : vxl_index + subj_n_vxl]
    test_3D_probas[subj_idx][mask_test[subj_idx]] = subj_3d_probas
    subj_3D_y = y_test[vxl_index : vxl_index + subj_n_vxl]
    test_3D_y[subj_idx][mask_test[subj_idx]] = subj_3D_y
    vxl_index += subj_n_vxl

## Plot test outputs and GT
output_dataset = np.concatenate((test_3D_probas, test_3D_y.reshape(-1, n_x, n_y, n_z, 1)), axis=-1)
channel_names = ['0', '1', 'GT']
visualize_dataset(output_dataset, channel_names, experiment_save_dir, subject_ids=None, save_name='output_visualisation')

