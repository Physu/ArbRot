from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .inference_doublehead_moco import inference_segmentor_doublehead_moco, init_segmentor_doublehead_moco  # new modified
from .test import multi_gpu_test, single_gpu_test
from .train import get_root_logger, set_random_seed, train_segmentor, train_segmentor_policy
from .test_classification import multi_gpu_test_classification, single_gpu_test_classification

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor',
    'inference_segmentor', 'multi_gpu_test', 'single_gpu_test',
    'show_result_pyplot', 'multi_gpu_test_classification',
    'single_gpu_test_classification',
    'inference_segmentor_doublehead_moco', 'init_segmentor_doublehead_moco',
    'train_segmentor_policy'
]
