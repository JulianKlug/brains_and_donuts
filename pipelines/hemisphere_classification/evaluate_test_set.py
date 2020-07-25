import sys, os
sys.path.insert(0, '../../')
import analysis_tools.data_loader as dl
from topological_hemisphere_classification import evaluate_topological_hemisphere_classification


## Import data
# data_path = '/home/klug/working_data/angio_hemispheres/withAngio_hemispheres_all_2016_2017.npz'
# save_dir = '/home/klug/output/bnd/stroke_detection_grid_search'
data_path = '/Users/julian/stroke_research/brain_and_donuts/full_datasets/hemispheres_withAngio_all_2016_2017.npz'
save_dir = '/Users/julian/temp/bnd_output'
n_subjects = None
n_threads = 50
test_set_seed = 42

experiment_name = 'tree_best_val_test_run'

# Load data
data_dir = os.path.dirname(data_path)
data_set_name = os.path.basename(data_path)
clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params = \
    dl.load_structured_data(data_dir, data_set_name)

params = {
    'homology_dimensions': [0, 1, 2],
    'features': ['PersistenceEntropy', 'Amplitude', 'NumberOfPoints'],
    'amplitude_metric': 'betti',
    'processing_filter': False,
    'processing_scale': True,
    'processing_scaler_metric': 'landscape',
    'inverse_input': False,
    'subsampling_factor': 2,
    'model': 'RandomForestClassifier'
}

train_acc, test_acc, n_features, feature_creation_timing, feature_classification_and_prediction_time = evaluate_topological_hemisphere_classification(
                    ct_inputs, ct_lesion_GT, brain_masks, save_dir, experiment_name,
                    features = params['features'],
                    amplitude_metric = params['amplitude_metric'],
                    processing_filter = params['processing_filter'], processing_scale = params['processing_scale'], processing_scaler_metric = params['processing_scaler_metric'],
                    homology_dimensions=params['homology_dimensions'], inverse_input = params['inverse_input'], model = params['model'],
                    n_subjects = n_subjects, n_threads = n_threads, subsampling_factor = params['subsampling_factor'], verbose = True,
                    save_input_features = False, save_output = False, split_seed=test_set_seed
                )

print(f'Train Accuracy: {train_acc}')
print(f'Test Accuracy: {test_acc}')
print(f'n_features: {n_features}')
print(f'feature_creation_timing: {feature_creation_timing}')
print(f'feature_classification_and_prediction_time: {feature_classification_and_prediction_time}')