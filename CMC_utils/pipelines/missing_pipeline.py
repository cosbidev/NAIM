import os
import logging

from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

from .routines import initialize_experiment
from CMC_utils import miscellaneous, save_load, preprocessing, metrics, cross_validation as cv

log = logging.getLogger(__name__)

__all__ = ["supervised_tabular_missing_main"]


def supervised_tabular_missing_main(cfg: DictConfig) -> None:
    log.info("Supervised tabular missing main started")

    initialize_experiment(cfg)

    dataset = instantiate(cfg.db, model_label_types=cfg.model.label_types, model_framework=cfg.model.framework, preprocessing_params=cfg.preprocessing, _recursive_=False)

    cv.set_cross_validation(dataset.info_for_cv, cfg.paths.cv, test_params=cfg.test_cv, val_params=cfg.val_cv)

    for test_fold, val_fold, train, val, test, last_val_fold in cv.get_cross_validation(cfg.paths.cv, "train", "val", "test"):

        train_data, train_labels, val_data, val_labels = cv.get_sets_with_idx(dataset.data, train, val, labels=dataset.labels_for_model)

        for train_missing_fraction in cfg.missing_percentages:
            train_missing_percentage = int(100 * train_missing_fraction)

            train_data_missing, _ = preprocessing.generate_missing( train_data, **miscellaneous.recursive_cfg_substitute(cfg.missing_generation.train, {"missing_fraction": train_missing_fraction}) )
            val_data_missing, _ = preprocessing.generate_missing( val_data, **miscellaneous.recursive_cfg_substitute(cfg.missing_generation.val, {"missing_fraction": train_missing_fraction}) )

            preprocessing_paths = {key: os.path.join(value, str(train_missing_percentage)) for key, value in cfg.paths.preprocessing.items()}
            save_load.create_directories(**preprocessing_paths)

            train_set = instantiate(cfg.db.dataset_class, train_data_missing, train_labels, "train", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=preprocessing_paths, test_fold=test_fold, val_fold=val_fold, augmentation=cfg.model.name == "naim")
            val_set = instantiate(cfg.db.dataset_class, val_data_missing, val_labels, "val", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=preprocessing_paths, test_fold=test_fold, val_fold=val_fold)

            model_params = call(cfg.model.set_params_function, OmegaConf.to_object(cfg.model), preprocessing_params=dataset.preprocessing_params, train_set=train_set, val_set=val_set)
            model = instantiate(model_params["init_params"], _recursive_=False)

            model_path = os.path.join(cfg.paths.model, str(train_missing_percentage))
            train_params = OmegaConf.to_object(cfg.train)
            train_params["set_metrics"] = metrics.set_metrics_params(train_params.get("set_metrics", {}), preprocessing_params=dataset.preprocessing_params)
            call(model_params["train_function"], model, train_set, model_params, model_path, val_set=val_set, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

            prediction_path = os.path.join(cfg.paths.predictions, str(train_missing_percentage))
            save_load.create_directory(prediction_path)
            call(model_params["test_function"], train_set, val_set, model_params=model_params, model_path=model_path, prediction_path=prediction_path, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

            test_data, test_labels = cv.get_sets_with_idx(dataset.data, test, labels=dataset.labels_for_model)

            for test_missing_fraction in cfg.missing_percentages:
                test_missing_percentage = int(100 * test_missing_fraction)
                log.info(f"Test missing percentage: {test_missing_percentage}")

                test_data_missing, _ = preprocessing.generate_missing( test_data, **miscellaneous.recursive_cfg_substitute(cfg.missing_generation.test, {"missing_fraction": test_missing_fraction}) )
                test_set = instantiate(cfg.db.dataset_class, test_data_missing, test_labels, "test", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=preprocessing_paths, test_fold=test_fold, val_fold=val_fold)

                test_prediction_path = os.path.join(prediction_path, str(test_missing_percentage))
                save_load.create_directory(test_prediction_path)
                call(model_params["test_function"], test_set, model_params=model_params, model_path=model_path, prediction_path=test_prediction_path, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

                del test_data_missing, test_set

            del train_data_missing, val_data_missing, test_data, test_labels, train_set, val_set, model_params, train_params

        del train_data, train_labels, val_data, val_labels

    performance_metrics = metrics.set_metrics_params(cfg.performance_metrics, preprocessing_params=dataset.preprocessing_params)
    metrics.compute_missing_performance(dataset.classes, cfg.paths.predictions, cfg.paths.results, performance_metrics, cfg.missing_percentages)

    log.info("Job finished")


if __name__ == "__main__":
    pass
