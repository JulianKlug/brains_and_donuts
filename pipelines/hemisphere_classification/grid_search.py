import sys, os
sys.path.insert(0, '../../')
from analysis_tools.utils.utils import create_experiment_name
import analysis_tools.data_loader as dl
from topological_hemisphere_classification import evaluate_topological_hemisphere_classification
from sklearn.model_selection import ParameterGrid, train_test_split
from tqdm import tqdm
import pandas as pd


## Import data
data_path = '/home/klug/working_data/angio_hemispheres/withAngio_hemispheres_all_2016_2017.npz'
save_dir = '/home/klug/output/bnd/stroke_detection_grid_search'
# data_path = '/Users/julian/stroke_research/brain_and_donuts/full_datasets/hemispheres_withAngio_all_2016_2017.npz'
# save_dir = '/Users/julian/temp/bnd_output'
result_file = 'grid_search.csv'
n_subjects = None
n_threads = 50
test_set_seed = 42
result_path = os.path.join(save_dir, result_file)

# Load data
data_dir = os.path.dirname(data_path)
data_set_name = os.path.basename(data_path)
clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params = \
    dl.load_structured_data(data_dir, data_set_name)

# Do not look at test set
X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(ct_inputs, ct_lesion_GT, brain_masks, test_size=0.2, random_state=test_set_seed)

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


for params in tqdm(list(ParameterGrid(params_search_space))):
        print(params)
        experiment_name = create_experiment_name(params, mode='readable')
        experiment_save_name = create_experiment_name(params, mode='readable', save_dir=save_dir)

        if os.path.exists(result_path):
            all_results = pd.read_csv(result_path)
            if experiment_name in all_results['experiment_name'].values:
                print('Experiment already done.')
                continue

        processing_scale = True
        if params['processing_scaler_metric'] == 'None':
            processing_scale = False

        try:
            train_acc, validation_acc, n_features, feature_creation_timing, feature_classification_and_prediction_time = evaluate_topological_hemisphere_classification(
                    X_train, y_train, mask_train, save_dir, experiment_save_name,
                    features = params['features'],
                    amplitude_metric = params['amplitude_metric'],
                    processing_filter = params['processing_filter'], processing_scale = processing_scale, processing_scaler_metric = params['processing_scaler_metric'],
                    homology_dimensions=params['homology_dimensions'], inverse_input = params['inverse_input'],
                    n_subjects = n_subjects, n_threads = n_threads, subsampling_factor = params['subsampling_factor'], verbose = True,
                    save_input_features = False, save_output = False, model = params['model']
                )

            experiment_results = pd.DataFrame([[]])
            experiment_results['experiment_name'] = experiment_name
            experiment_results['train_accuracy'] = train_acc
            experiment_results['validation_accuracy'] = validation_acc
            experiment_results['n_features'] = n_features
            experiment_results['feature_creation_timing'] = feature_creation_timing
            experiment_results['feature_classification_and_prediction_time'] = feature_classification_and_prediction_time
            for key, value in zip(params.keys(), params.values()):
                 experiment_results[key] = str(value)
            experiment_results['test_set_seed'] = test_set_seed

            if not os.path.exists(result_path):
                experiment_results.to_csv(result_path, index=False)
            else:
                combined_results = pd.concat([all_results, experiment_results])
                combined_results.to_csv(result_path, index=False)
        except Exception as e:
            print('FAILED')
            print('Used params:', params)
            print(e)



