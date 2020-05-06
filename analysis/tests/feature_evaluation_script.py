import sys, os, time

path_bnd = '../..'
sys.path.insert(1, path_bnd)
import analysis.data_loader as dl
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import CubicalPersistence
from pgtda.diagrams import PersistenceEntropy, Amplitude, Filtering, Scaler
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pgtda.images import RollingSubImageTransformer, make_image_union
from analysis.utils.plot_ROC import plot_roc
from analysis.utils.metrics import dice, roc_auc
from analysis.utils.dataset_visualization import visualize_dataset

## Import data
data_dir = '/Users/julian/stroke_research/brain_and_donuts/full_datasets'
save_dir = '/Users/julian/temp/bnd_pipe_test'
data_set_name = 'withAngio_all_2016_2017.npz'
model_name = 'CubicalPersistance_With_PersistenceEntropy_And_WassersteinAmplitude'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params = \
    dl.load_structured_data(data_dir, data_set_name)

# Reshape ct_inputs as it has 1 channel
ct_inputs = ct_inputs.reshape((*ct_inputs.shape[:-1]))

n_images = 1
X = (ct_inputs[:n_images] * brain_masks[:n_images])[range(n_images), ::2, ::2, ::2]
y = (ct_lesion_GT[:n_images] * brain_masks[:n_images])[range(n_images), ::2, ::2, ::2]

## Feature Creation
width_list = [[5, 5, 5], [7, 7, 7]]
start = time.time()
# Note that padding should be same so that output images always have the same size
transformer = make_pipeline(CubicalPersistence(homology_dimensions=(0, 1 ,2), n_jobs=1), Filtering(epsilon=np.max(X)-1, below=False), Scaler(),
                             make_union(PersistenceEntropy(n_jobs=1),
                                         Amplitude(metric='wasserstein', metric_params={'p':2}, order=None, n_jobs=1)))
rsis = make_image_union(*[RollingSubImageTransformer(transformer=transformer, width=width, padding='same')
                    for width in width_list], n_jobs=1)
X_features = rsis.fit_transform(X)
end = time.time()
feature_creation_timing = end - start

## Feature Classification
#### Create classifier
start = time.time()
classifier = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
#### Prepare dataset

n_images, n_x, n_y, n_z, n_features = X_features.shape
X_flat = X_features.reshape(n_images, -1, n_features)
y_flat = y.reshape(n_images, -1)

X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.3, random_state=42)
X_train, y_train = X_train.reshape(-1, n_features), y_train.reshape(-1)
X_test, y_test = X_test.reshape(-1, n_features), y_test.reshape(-1)

#### Train classifier
classifier.fit(X_train, y_train)

#### Apply classifier
probas = classifier.predict_proba(X_test)
predicted = classifier.predict(X_test)

#### Reconstruct output
probas_3D = probas.reshape(-1, n_x, n_y, n_z, 2)
predicted_3D = predicted.reshape(-1, n_x, n_y, n_z)
end = time.time()
feature_classification_and_prediction_time = end - start

## Model (Features + Classifier) Evaluation
dice_score = dice(predicted.flatten(), y_test.flatten())
roc_auc_score, roc_curve_details = roc_auc(y_test, predicted)

print('Dice:', dice_score)
print('ROC AUC:', roc_auc_score)

with open(os.path.join(save_dir, 'logs.txt', "a")) as log_file:
    log_file.write('Dice: %s\n' % dice_score)
    log_file.write('ROC AUC: %s\n' % roc_auc_score)
    log_file.write('Feature Creation timing: %s\n' % feature_creation_timing)
    log_file.write('Feature Classification and Prediction timing: %s\n' % feature_classification_and_prediction_time)

# %%
fpr, tpr, roc_thresholds = roc_curve_details
plot_roc([tpr], [fpr], save_dir=save_dir, save_plot=True, model_name=model_name)

## Model feature analysis
#### Model confusion matrix
confusion = confusion_matrix(y_test, predicted)
plt.imshow(confusion)
plt.savefig(os.path.join(save_dir, model_name + '_confusion_matrix.png'))
#### Feature correalation
correlation = np.abs(np.corrcoef(X_train.T))
plt.imshow(correlation)
plt.savefig(os.path.join(save_dir, model_name + '_correlation_matrix.png'))

## Plot test outputs and GT
output_dataset = np.concatenate((probas_3D, y_test.reshape(-1, n_x, n_y, n_z, 1)), axis=-1)
channel_names = ['0', '1', 'GT']
visualize_dataset(output_dataset, channel_names, save_dir, subject_ids=None, save_name='output_visualisation')

