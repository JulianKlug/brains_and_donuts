import sys, os, time, pickle

path_bnd = '../../'
sys.path.insert(1, path_bnd)
from analysis_tools.utils.utils import invert_image
import numpy as np
from gtda.homology import CubicalPersistence
from pgtda.diagrams import PersistenceEntropy, Amplitude, Filtering, Scaler, NumberOfPoints
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def evaluate_topological_hemisphere_classification(
        X, y, x_mask, save_dir, experiment_name,
        features = ['PersistenceEntropy', 'Amplitude','NumberOfPoints'],
        amplitude_metric = 'wasserstein',
        processing_filter = True, processing_scale = True, processing_scaler_metric = 'bottleneck',
        homology_dimensions=(0, 1 ,2), inverse_input = True,
        n_subjects = None, n_threads = 50, subsampling_factor = 2, verbose = True,
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

    lesion_presence_GT = np.any(y, axis=(1,2,3))

    # Reshape ct_inputs as it has 1 channel
    X = X.reshape((*X.shape[:-1]))

    if n_subjects is None:
        n_subjects = X.shape[0]

    # Apply brain masks
    X = (X[:n_subjects] * x_mask[:n_subjects])[range(n_subjects), ::subsampling_factor, ::subsampling_factor, ::subsampling_factor]
    y = lesion_presence_GT[:n_subjects]

    # Normalise data
    # Capping (threshold to 0-500 as values outside this range seem non relevant to the vascular analysis)
    vmin = 0
    vmax = 500
    X[X < vmin] = vmin
    X[X > vmax] = vmax

    ## Feature Creation
    n_widths_for_thread_attribution = 1 # TODO verify sklearn jobs
    start = time.time()

    if inverse_input:
        X = invert_image(X)

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

    X_features = transformer.fit_transform(X)

    n_features = X_features.shape[1]

    end = time.time()
    feature_creation_timing = end - start
    if verbose:
        print(f'Features ready after {feature_creation_timing} s')

    ## Feature Classification
    #### Create classifierå
    start = time.time()
    classifier = RandomForestClassifier(n_estimators=10000, n_jobs=-1)

    #### Prepare dataset
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)

    if save_input_features:
        pickle.dump(X_train, open(os.path.join(pickle_dir, 'X_train.p'), 'wb'))
        pickle.dump(X_test, open(os.path.join(pickle_dir, 'X_test.p'), 'wb'))
        pickle.dump(y_train, open(os.path.join(pickle_dir, 'y_train.p'), 'wb'))
        pickle.dump(y_test, open(os.path.join(pickle_dir, 'y_test.p'), 'wb'))

    #### Train classifier
    classifier.fit(X_train, y_train)

    #### Apply classifier
    test_probas = classifier.predict_proba(X_test)
    test_predicted = classifier.predict(X_test)

    train_probas = classifier.predict_proba(X_train)
    train_predicted = classifier.predict(X_train)

    if save_output:
        ### save classifier
        pickle.dump(classifier, open(os.path.join(pickle_dir, 'trained_classifier.p'), 'wb'))

        ### save predicted output
        pickle.dump(test_probas, open(os.path.join(pickle_dir, 'test_probas.p'), 'wb'))
        pickle.dump(test_predicted, open(os.path.join(pickle_dir, 'test_predicted.p'), 'wb'))

        pickle.dump(train_probas, open(os.path.join(pickle_dir, 'train_probas.p'), 'wb'))
        pickle.dump(train_predicted, open(os.path.join(pickle_dir, 'train_predicted.p'), 'wb'))

    #### Reconstruct output
    end = time.time()
    feature_classification_and_prediction_time = end - start

    ## Model (Features + Classifier) Evaluation
    train_acc = accuracy_score(train_predicted, y_train)
    test_acc = accuracy_score(test_predicted, y_test)

    if verbose:
        print('Train Accuracy:', train_acc)
        print('Test Accuracy:', test_acc)

    with open(os.path.join(experiment_save_dir, 'logs.txt'), "a") as log_file:
        log_file.write('Train Accuracy: %s\n' % train_acc)
        log_file.write('Test Accuracy: %s\n' % test_acc)
        log_file.write('Feature Creation timing: %s\n' % feature_creation_timing)
        log_file.write('Feature Classification and Prediction timing: %s\n' % feature_classification_and_prediction_time)

    ## Model feature analysis
    #### Model confusion matrix
    # confusion = confusion_matrix(y_test, test_predicted)
    # plt.imshow(confusion)
    # plt.savefig(os.path.join(experiment_save_dir, experiment_name + '_confusion_matrix.png'))
    #
    # #### Feature correlation
    # correlation = np.abs(np.corrcoef(X_train.T))
    # plt.imshow(correlation)
    # plt.savefig(os.path.join(experiment_save_dir, experiment_name + '_correlation_matrix.png'))

    return train_acc, test_acc, n_features, feature_creation_timing, feature_classification_and_prediction_time
