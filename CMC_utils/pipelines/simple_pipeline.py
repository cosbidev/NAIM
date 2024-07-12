import logging
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
from .routines import initialize_experiment
from CMC_utils import metrics
from CMC_utils import cross_validation as cv

log = logging.getLogger(__name__)

__all__ = ["supervised_learning_main"]


def supervised_learning_main(cfg: DictConfig) -> None:
    log.info(f"Supervised main started")

    initialize_experiment(cfg)

    dataset = instantiate(cfg.db, model_label_types=cfg.model.label_types, model_framework=cfg.model.framework, preprocessing_params=cfg.preprocessing, _recursive_=False)

    cv.set_cross_validation(dataset.info_for_cv, cfg.paths.cv, test_params=cfg.test_cv, val_params=cfg.val_cv)

    for test_fold, val_fold, train, val, test, last_val_fold in cv.get_cross_validation(cfg.paths.cv, "train", "val", "test"):
        train_data, train_labels, val_data, val_labels = cv.get_sets_with_idx(dataset.data, train, val, labels=dataset.labels_for_model)

        train_set = instantiate(cfg.db.dataset_class, train_data, train_labels, "train", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=cfg.paths.preprocessing, test_fold=test_fold, val_fold=val_fold, augmentation=cfg.model.name == "naim")
        val_set = instantiate(cfg.db.dataset_class, val_data, val_labels, "val", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=cfg.paths.preprocessing, test_fold=test_fold, val_fold=val_fold)

        model_params = call(cfg.model.set_params_function, OmegaConf.to_object(cfg.model), preprocessing_params=dataset.preprocessing_params, train_set=train_set, val_set=val_set)
        model = instantiate(model_params["init_params"], _recursive_=False)

        train_params = OmegaConf.to_object(cfg.train)
        train_params["set_metrics"] = metrics.set_metrics_params(train_params.get("set_metrics", {}), preprocessing_params=dataset.preprocessing_params)
        call(model_params["train_function"], model, train_set, model_params, cfg.paths.model, val_set=val_set, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

        call(model_params["test_function"], train_set, val_set, model_params=model_params, model_path=cfg.paths.model, prediction_path=cfg.paths.predictions, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

        test_data, test_labels = cv.get_sets_with_idx(dataset.data, test, labels=dataset.labels_for_model)
        test_set = instantiate(cfg.db.dataset_class, test_data, test_labels, "test", preprocessing_params=dataset.preprocessing_params, preprocessing_paths=cfg.paths.preprocessing, test_fold=test_fold, val_fold=val_fold)

        call(model_params["test_function"], test_set, model_params=model_params, model_path=cfg.paths.model, prediction_path=cfg.paths.predictions, classes=dataset.classes, train_params=train_params, test_fold=test_fold, val_fold=val_fold, _recursive_=False)

        del train_data, train_labels, val_data, val_labels, test_data, test_labels, train_set, val_set, test_set, model_params, train_params

    performance_metrics = metrics.set_metrics_params(cfg.performance_metrics, preprocessing_params=dataset.preprocessing_params)
    metrics.compute_performance(dataset.classes, cfg.paths.predictions, cfg.paths.results, performance_metrics)

    log.info(f"Job finished")


if __name__ == "__main__":
    pass
