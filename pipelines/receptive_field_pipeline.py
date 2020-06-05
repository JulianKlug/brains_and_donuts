import sys, os, time, pickle

path_bnd = '../'
sys.path.insert(1, path_bnd)
import analysis_tools.data_loader as dl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pgtda.images import RollingSubImageTransformer
from analysis_tools.metrics.plot_ROC import plot_roc
from analysis_tools.metrics.metrics import dice, roc_auc
from visual_tools.dataset_visualization import visualize_dataset

## Import data
data_dir = '/media/miplab-nas2/Data/klug/geneva_stroke_dataset/working_data/withAngio_all_2016_2017'
save_dir = '/home/klug/output/bnd/feature_eval'
data_set_name = 'data_set.npz'
experiment_name = 'rf_base_w5'

n_images = 113
n_threads = 50
subsampling_factor = 1
batch_size = 10

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
X = (ct_inputs[:n_images] * brain_masks[:n_images])[range(n_images), ::subsampling_factor, ::subsampling_factor, ::subsampling_factor]
y = (ct_lesion_GT[:n_images] * brain_masks[:n_images])[range(n_images), ::subsampling_factor, ::subsampling_factor, ::subsampling_factor]

# Normalise data
# Capping (threshold to 0-500 as values outside this range seem non relevant to the vascular analysis)
vmin = 0
vmax = 500
X[X < vmin] = vmin
X[X > vmax] = vmax


## Feature Creation
width = [7, 7, 7]
n_widths = 1
start = time.time()
# Note that padding should be same so that output images always have the same size
rsis = RollingSubImageTransformer(width=width, padding='same')

# Batch decomposition to spare memory
X_features = None
for batch_offset in tqdm(range(0, X.shape[0], batch_size)):
    batch = X[batch_offset:batch_offset+batch_size]
    batch_features = rsis.fit_transform(batch)
    n_i, n_x, n_y, n_z = batch_features.shape[0:4]
    batch_features = batch_features.reshape(n_i, n_x, n_y, n_z, -1)
    if X_features is None:
        X_features = batch_features
    else:
        X_features = np.concatenate((X_features, batch_features), axis=0)

end = time.time()
feature_creation_timing = end - start
print(f'Features ready after {feature_creation_timing}s')

## Feature Classification
#### Create classifier
start = time.time()
classifier = RandomForestClassifier(n_estimators=100, n_jobs=n_threads)
#### Prepare dataset

n_images, n_x, n_y, n_z, n_features = X_features.shape
X_flat = X_features.reshape(n_images, -1, n_features)
y_flat = y.reshape(n_images, -1)

X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.3, random_state=42)
X_train, y_train = X_train.reshape(-1, n_features), y_train.reshape(-1)
X_test, y_test = X_test.reshape(-1, n_features), y_test.reshape(-1)

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
probas_3D = test_probas.reshape(-1, n_x, n_y, n_z, 2)
predicted_3D = test_predicted.reshape(-1, n_x, n_y, n_z)
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

## Plot test outputs and GT
output_dataset = np.concatenate((probas_3D, y_test.reshape(-1, n_x, n_y, n_z, 1)), axis=-1)
channel_names = ['0', '1', 'GT']
visualize_dataset(output_dataset, channel_names, experiment_save_dir, subject_ids=None, save_name='output_visualisation')

