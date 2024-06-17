import os
import hydra
import torch
import mlflow
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.optim import SGD, Adam
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from modeling import pipeline
from visualization.plot import Plotter
from modeling.train.training import Training, ContrastiveLoss
from modeling.train.input import InputPipeline
from consts.paths import CONFIG_PATH, OUTPUT_PATH, DatasetPaths
from consts.consts import DatasetSplit, SiameseArchitectures
from modeling.train.preprocess import SiamesePreprocess

@hydra.main(version_base=None, config_path=CONFIG_PATH)
def main(config):
    experiment_name = config.architecture.run.experiment_name
    arch = SiameseArchitectures(config.architecture.pipeline.architecture)
    batch_size = int(config.architecture.pipeline.batch_size)
    learning_rate = float(config.architecture.pipeline.learning_rate)
    epochs = int(config.architecture.pipeline.epochs)

    logger.info('Starting training process...')
    logger.info('------------------------------------')

    logger.info('Creating MLFlow experiment...')
    logger.info('------------------------------------')

    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(experiment_name)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    logger.info(f'CUDA available {use_cuda}')
    logger.info('------------------------------------')

    with mlflow.start_run(experiment_id=experiment_id) as run:
        artifact_path = os.path.join(
            OUTPUT_PATH, f'{experiment_id}', f'{run.info.run_id}', 'artifacts'
        )
        logger.add(os.path.join(artifact_path, 'train_log.log'))

        logger.info('MLFlow session started...')
        logger.info('------------------------------------')
        logger.info(f'Run: {run.info.run_id}')
        logger.info(f'Experiment ID: {experiment_id}')
        logger.info(f'Experiment name {experiment_name}')

        metadata_csv = DatasetPaths.processed_dataset_metadata
        train_ds = SiamesePreprocess(metadata_csv, split="train", 
                                    processed_dataset=DatasetPaths.processed_dataset)
        val_ds = SiamesePreprocess(metadata_csv, split='val',
                                processed_dataset=DatasetPaths.processed_dataset)

        input_pipeline = InputPipeline(
            datasets_list=[train_ds, val_ds],
            batch_size=batch_size,
            pin_memory=True if use_cuda else False,
        )

        n_outputs = train_ds.get_n_outputs()
        logger.info(f'Number of classes: {n_outputs}')
        logger.info('------------------------------------')

        model = pipeline.SIAMESE_ARCHITECTURE[arch]()

        # Hyperparameter tuning
        param_grid = {
            'lr': [0.0005, 0.001, 0.01],
            'batch_size': [32, 64, 128],
        }
        logger.info(f'Parameter Grid: {param_grid}')
        logger.info('------------------------------------')

        logger.info(f'Extracting Data from the input pipeline...')

        # Extract data and labels from the input pipeline
        X_train, y_train = [], []
        for batch in tqdm(input_pipeline[DatasetSplit.TRAIN]):
            selfie_images, idcard_images, labels = batch
            X_train.append((selfie_images, idcard_images))
            y_train.append(labels)

        # Combine all data and labels
        selfie_images_combined = torch.cat([x[0] for x in X_train], dim=0)
        idcard_images_combined = torch.cat([x[1] for x in X_train], dim=0)
        y_train = torch.cat(y_train, dim=0)
        X_train_combined = torch.cat([selfie_images_combined, idcard_images_combined], dim=1)


        logger.info("Hyperparameter Tuning...")

        model_hyperparameters = NeuralNetClassifier(
            module=model,
            max_epochs=150,
            batch_size=10
        )

        # Specify the cross-validation strategy
        cv_strategy = LeaveOneOut()

        # Set up GridSearchCV object
        grid_search = GridSearchCV(
            estimator=model_hyperparameters,
            param_grid=param_grid,
            scoring=make_scorer(accuracy_score),
            cv=cv_strategy,
            n_jobs=-1,
        )


        logger.info("Grid Search...")
        grid_search.fit(X_train_combined, y_train)

        # Get the best hyperparameters and the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Set hyperparameters based on the best_params
        learning_rate = best_params['learning_rate']
        batch_size = best_params['batch_size']

        # Log best hyperparameters
        logger.info(f'Best Hyperparameters: {best_params}')
        mlflow.log_params(best_params)

        loss_function = ContrastiveLoss(margin=2.0).to(device)
        optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
        )

        logger.info("Starting to train...")
        training = Training(
            input_pipeline=input_pipeline,
            model=best_model,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            early_stop=True,
        )

        if use_cuda:
            torch.backends.cudnn.benchmark = True

        logger.info(f'Epochs: {epochs}')
        mlflow.log_param('epochs', epochs)
        logger.info(f'Batch size: {batch_size}')
        mlflow.log_param('batch_size', batch_size)
        logger.info(f'Learning rate: {learning_rate}')
        mlflow.log_param('learning_rate', learning_rate)
        logger.info('------------------------------------')
        model, train_accuracy, val_accuracy, train_loss, val_loss = training.train(
            epochs
        )

        logger.info('Model metrics')
        logger.info(f'Training accuracy: {train_accuracy[-1:][0]}')
        mlflow.log_metric('training_accuracy', train_accuracy[-1:][0])
        logger.info(f'Training loss: {train_loss[-1:][0]}')
        mlflow.log_metric('training_loss', train_loss[0])
        logger.info(f'Validation accuracy: {val_accuracy[-1:][0]}')
        mlflow.log_metric('validation_accuracy', val_accuracy[-1:][0])
        logger.info(f'Validation loss: {val_loss[-1:][0]}')
        mlflow.log_metric('validation_loss', val_loss[-1:][0])
        logger.info('------------------------------------')
        logger.info(f'Average training accuracy: {np.mean(train_accuracy)}')
        mlflow.log_metric('average_training_accuracy', np.mean(train_accuracy))
        logger.info(f'Average training loss: {np.mean(train_loss)}')
        mlflow.log_metric('average_training_loss', np.mean(train_loss))
        logger.info(f'Average validation accuracy: {np.mean(val_accuracy)}')
        mlflow.log_metric('average_validation_accuracy', np.mean(val_accuracy))
        logger.info(f'Average validation loss: {np.mean(val_loss)}')
        mlflow.log_metric('average_validation_loss', np.mean(val_loss))
        logger.info('------------------------------------')

        logger.info(f'Plotting metrics...')
        plotter = Plotter()
        acc_plot_path = os.path.join(artifact_path, 'accuracy_train.png')
        loss_plot_path = os.path.join(artifact_path, 'loss_train.png')

        plotter.plot_accuracy(train_accuracy, val_accuracy, acc_plot_path, arch)
        plotter.plot_loss(train_loss, val_loss, loss_plot_path, arch)
        logger.info(f'Logging plots to {artifact_path}')
        logger.info('------------------------------------')

        logger.info(f'Logging PyTorch model to {artifact_path}')
        mlflow.pytorch.log_model(model, 'model')
        logger.info('------------------------------------')

if __name__ == '__main__':
    main()
