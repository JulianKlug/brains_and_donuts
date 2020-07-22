import sys, os
sys.path.insert(0, '../../')
from analysis_tools.utils.utils import combiset, create_experiment_name
from topological_hemisphere_classification import evaluate_topological_hemisphere_classification
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import pandas as pd


## Import data
# data_dir = '/media/miplab-nas2/Data/klug/geneva_stroke_dataset/working_data/withAngio_all_2016_2017'
# save_dir = '/home/klug/output/bnd/feature_eval'
data_path = '/Users/julian/stroke_research/brain_and_donuts/full_datasets/withAngio_hemispheres_all_2016_2017.npz'
save_dir = '/Users/julian/temp/bnd_output'
result_file = 'grid_search.csv'
result_path = os.path.join(save_dir, result_file)

params_search_space = {
    'homology_dimensions': combiset([0, 1, 2]),
    'features': combiset(['PersistenceEntropy', 'Amplitude','NumberOfPoints']),
    # 'amplitude_metric': ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat', 'silhouette', 'persistence_image', 'landscape'],
    'amplitude_metric': ['wasserstein'],
    'processing_filter': [True, False],
    'processing_scale': [True, False],
    # 'processing_scaler_metric': ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat', 'silhouette', 'persistence_image', 'landscape'],
    'processing_scaler_metric': ['bottleneck'],
    'inverse_input': [True, False],
    'subsampling_factor': [1, 2, 3]
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

        try:
            train_acc, test_acc, n_features, feature_creation_timing, feature_classification_and_prediction_time = evaluate_topological_hemisphere_classification(
                    data_path, save_dir, experiment_save_name,
                    features = params['features'],
                    amplitude_metric = params['amplitude_metric'],
                    processing_filter = params['processing_filter'], processing_scale = params['processing_scale'], processing_scaler_metric = params['processing_scaler_metric'],
                    homology_dimensions=params['homology_dimensions'], inverse_input = params['inverse_input'],
                    n_subjects = 3, n_threads = 50, subsampling_factor = params['subsampling_factor'], verbose = True,
                    save_input_features = False, save_output = False
                )

            experiment_results = pd.DataFrame([[]])
            experiment_results['experiment_name'] = experiment_name
            experiment_results['train_accuracy'] = train_acc
            experiment_results['test_accuracy'] = test_acc
            experiment_results['n_features'] = n_features
            experiment_results['feature_creation_timing'] = feature_creation_timing
            experiment_results['feature_classification_and_prediction_time'] = feature_classification_and_prediction_time
            for key, value in zip(params.keys(), params.values()):
                 experiment_results[key] = value

            if not os.path.exists(result_path):
                experiment_results.to_csv(result_path, index=False)
            else:
                combined_results = pd.concat([all_results, experiment_results])
                combined_results.to_csv(result_path, index=False)
        except:
            print('FAILED')
            print('Used params:', params)


