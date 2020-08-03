import sys, os, time, pickle

path_bnd = '../../'
sys.path.insert(1, path_bnd)
from analysis_tools.utils.utils import invert_image
from pipelines.patch_classification.brain_segmentation_to_subimages_lesion_presence import brain_segmentation_to_subimages_lesion_presence
import numpy as np
from gtda.homology import CubicalPersistence
from pgtda.diagrams import PersistenceEntropy, Amplitude, Filtering, Scaler, NumberOfPoints
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.under_sampling import NearMiss
import analysis_tools.data_loader as dl


def evaluate_topological_3D_patch_classification(
        X, y, x_mask, save_dir, experiment_name,
        features = ['PersistenceEntropy', 'Amplitude','NumberOfPoints'],
        amplitude_metric = 'wasserstein',
        processing_filter = True, processing_scale = True, processing_scaler_metric = 'bottleneck',
        homology_dimensions=(0, 1 ,2), inverse_input = True, width = 7,
        model='LogisticRegression', undersampling = True,
        n_subjects = None, n_threads = 50, subsampling_factor = 2, split_seed=42, verbose = True,
        save_input_features = False, save_output = False
    ):

    # Create necessary directories to save data
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    experiment_save_dir = os.path.join(save_dir, experiment_name)
    if not os.path.exists(experiment_save_dir):
        os.mkdir(experiment_save_dir)
    else:
        os.mkdir(f'{experiment_save_dir}_{time.strftime("%Y%m%d_%H%M%S")}')

    if save_output or save_input_features:
        pickle_dir = os.path.join(experiment_save_dir, 'pickled_data')
        if not os.path.exists(pickle_dir):
            os.mkdir(pickle_dir)

    # Reshape ct_inputs as it has 1 channel
    X = X.reshape((*X.shape[:-1]))

    if n_subjects is None:
        n_subjects = X.shape[0]

    # Apply brain masks
    X = (X[:n_subjects] * x_mask[:n_subjects])[range(n_subjects), ::subsampling_factor, ::subsampling_factor, ::subsampling_factor]
    y = (y[:n_subjects] * x_mask[:n_subjects])[range(n_subjects), ::subsampling_factor, ::subsampling_factor, ::subsampling_factor]
    masks = x_mask[range(n_subjects), ::subsampling_factor, ::subsampling_factor, ::subsampling_factor]

    # Normalise data
    # Capping (threshold to 0-500 as values outside this range seem non relevant to the vascular analysis)
    vmin = 0
    vmax = 500
    X[X < vmin] = vmin
    X[X > vmax] = vmax

    start_subimage_creation = time.time()

    if inverse_input:
        X = invert_image(X)

    # Subimage creation
    patch_X, patch_y = brain_segmentation_to_subimages_lesion_presence(X, y, masks, width=width)

    # split subject wise
    X_train, X_validation, y_train, y_validation = train_test_split(patch_X, patch_y, test_size=0.3, random_state=split_seed)
    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    X_validation, y_validation = np.concatenate(X_validation), np.concatenate(y_validation)
    X_validation, X_train = X_validation.reshape(X_validation.shape[0], -1), X_train.reshape(X_train.shape[0], -1)

    # Subimage undersampling
    if undersampling:
        undersampler = NearMiss(version=3)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

    end_subimage_creation = time.time()
    subimage_creation_timing = end_subimage_creation - start_subimage_creation

    if verbose:
        print('Train data:', X_train.shape, y_train.shape)
        print('Validation data:', X_validation.shape, y_validation.shape)
        print(f'Subimages ready in {subimage_creation_timing} seconds.')

    ## Feature Creation
    n_widths_for_thread_attribution = 1  # TODO verify sklearn jobs
    start_feature_creation = time.time()

    feature_transformers = []
    if 'PersistenceEntropy' in features:
        feature_transformers.append(PersistenceEntropy(n_jobs=1))
    if 'Amplitude' in features:
        feature_transformers.append(Amplitude(metric=amplitude_metric, order=None, n_jobs=1))
    if 'NumberOfPoints' in features:
        feature_transformers.append(NumberOfPoints(n_jobs=1))
    n_subimage_features = len(feature_transformers)

    processing_pipeline = [
                        CubicalPersistence(homology_dimensions=homology_dimensions, n_jobs=int(n_threads/n_widths_for_thread_attribution)),
                        make_union(*feature_transformers, n_jobs=int((n_threads/n_widths_for_thread_attribution)/n_subimage_features))
                    ]

    if processing_filter:
        processing_pipeline.insert(1, Filtering(epsilon=np.max(X) - 1, below=False))
    if processing_scale:
        processing_pipeline.insert(-1, Scaler(metric=processing_scaler_metric))

    transformer = make_pipeline(*processing_pipeline)

    X_train_features = transformer.fit_transform(X_train)
    X_validation_features = transformer.fit_transform(X_validation)

    n_features = X_train_features.shape[1]

    end_feature_creation = time.time()
    feature_creation_timing = end_feature_creation - start_feature_creation
    if verbose:
        print(f'Features ready after {feature_creation_timing} seconds.')

    ## Feature Classification
    #### Create classifier
    start_feature_classification = time.time()
    if model == 'LogisticRegression':
        classifier = LogisticRegression(n_jobs=-1)
    elif model == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
    else:
        raise Exception(f'Model {model} not known')

    if save_input_features:
        pickle.dump(X_train_features, open(os.path.join(pickle_dir, 'X_train.p'), 'wb'))
        pickle.dump(X_validation_features, open(os.path.join(pickle_dir, 'X_valid.p'), 'wb'))
        pickle.dump(y_train, open(os.path.join(pickle_dir, 'y_train.p'), 'wb'))
        pickle.dump(y_validation, open(os.path.join(pickle_dir, 'y_valid.p'), 'wb'))

    #### Train classifier
    classifier.fit(X_train_features, y_train)

    #### Apply classifier
    valid_probas = classifier.predict_proba(X_validation_features)
    valid_predicted = classifier.predict(X_validation_features)

    train_probas = classifier.predict_proba(X_train_features)
    train_predicted = classifier.predict(X_train_features)

    if save_output:
        ### save classifier
        pickle.dump(classifier, open(os.path.join(pickle_dir, 'trained_classifier.p'), 'wb'))

        ### save predicted output
        pickle.dump(valid_probas, open(os.path.join(pickle_dir, 'valid_probas.p'), 'wb'))
        pickle.dump(valid_predicted, open(os.path.join(pickle_dir, 'valid_predicted.p'), 'wb'))

        pickle.dump(train_probas, open(os.path.join(pickle_dir, 'train_probas.p'), 'wb'))
        pickle.dump(train_predicted, open(os.path.join(pickle_dir, 'train_predicted.p'), 'wb'))

    #### Reconstruct output
    end_feature_classification = time.time()
    feature_classification_and_prediction_time = end_feature_classification - start_feature_classification

    ## Model (Features + Classifier) Evaluation
    train_acc = accuracy_score(train_predicted, y_train)
    valid_acc = accuracy_score(valid_predicted, y_validation)

    train_roc_auc = roc_auc_score(y_train, train_probas[:, 1])
    valid_roc_auc = roc_auc_score(y_validation, valid_probas[:, 1])

    if verbose:
        print('Train Accuracy:', train_acc)
        print('Test Accuracy:', valid_acc)
        print('Train ROC AUC:', train_roc_auc)
        print('Test ROC AUC:', valid_roc_auc)

    with open(os.path.join(experiment_save_dir, 'logs.txt'), "a") as log_file:
        log_file.write('Train Accuracy: %s\n' % train_acc)
        log_file.write('Validation Accuracy: %s\n' % valid_acc)
        log_file.write('Train ROC AUC: %s\n' % train_roc_auc)
        log_file.write('Validation ROC AUC: %s\n' % valid_roc_auc)
        log_file.write('Subimage Creation timing: %s\n' % subimage_creation_timing)
        log_file.write('Feature Creation timing: %s\n' % feature_creation_timing)
        log_file.write('Feature Classification and Prediction timing: %s\n' % feature_classification_and_prediction_time)

    return train_acc, valid_acc, train_roc_auc, valid_roc_auc, n_features, subimage_creation_timing, feature_creation_timing, feature_classification_and_prediction_time
