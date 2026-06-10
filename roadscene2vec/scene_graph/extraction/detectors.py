"""Unified detector factory and wrappers for different object detectors.

Provide a simple, consistent interface so callers (like `RealExtractor`) can
initialize and use any detector through a small adapter.

Supported detectors: 'detectron2' and 'yolo'. Implementations do lazy imports
so the package only requires the detector backends you actually use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import torch


@dataclass
class DetectionResult:
    boxes: torch.Tensor  # Nx4 xyxy
    classes: torch.Tensor  # N
    image_size: Tuple[int, int]
    _plot_impl: Any = None

    def plot(self, image=None):
        if self._plot_impl is None:
            raise RuntimeError("No plot implementation available for this result")
        return self._plot_impl(image)


class BaseDetector:
    """Base detector adapter."""

    def __init__(self, settings: dict):
        self.settings = settings or {}

    @property
    def names(self) -> List[str]:
        raise NotImplementedError()

    def predict(self, image, **kwargs) -> DetectionResult:
        """Run detection on a single image and return a DetectionResult."""
        raise NotImplementedError()


class Detectron2Detector(BaseDetector):
    def __init__(self, settings: dict):
        super().__init__(settings)
        # lazy imports
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo
        from detectron2.data import MetadataCatalog
        from detectron2.utils import visualizer

        self._get_cfg = get_cfg
        self._DefaultPredictor = DefaultPredictor
        self._model_zoo = model_zoo
        self._MetadataCatalog = MetadataCatalog
        self._Visualizer = visualizer

        model_path = settings.get("model_path", 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(settings.get("confidence", 0.5))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        cfg.MODEL.DEVICE = settings.get("device", 'cpu')
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        # metadata for visualizer
        try:
            self._coco_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get('thing_classes')
        except Exception:
            self._coco_names = None

    @property
    def names(self) -> List[str]:
        return list(self._coco_names) if self._coco_names is not None else []

    def predict(self, image, **kwargs) -> DetectionResult:
        outputs = self.predictor(image)
        instances = outputs['instances'].to('cpu')
        boxes = instances.pred_boxes.tensor
        classes = instances.pred_classes
        image_size = instances.image_size

        def _plot(im):
            v = self._Visualizer.Visualizer(im[:, :, ::-1], self._MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(self.predictor(im)['instances'].to('cpu'))
            return out.get_image()[:, :, ::-1]

        return DetectionResult(boxes=boxes, classes=classes, image_size=image_size, _plot_impl=_plot)


class YOLODetector(BaseDetector):
    def __init__(self, settings: dict):
        super().__init__(settings)
        from ultralytics import YOLO

        model_path = settings.get("model_path", "yolov8n.pt")
        self.model = YOLO(model_path)
        names = self.model.names
        if isinstance(names, dict):
            self._names = [names[i] for i in sorted(names)]
        else:
            self._names = list(names)

    @property
    def names(self) -> List[str]:
        return self._names

    def predict(self, image, **kwargs) -> DetectionResult:
        # pass through confidence/iou/device if provided
        results = self.model.predict(image, verbose=False,
                                     conf=kwargs.get('conf', None),
                                     iou=kwargs.get('iou', None),
                                     device=kwargs.get('device', None))
        result = results[0]
        boxes = result.boxes.xyxy.cpu()
        classes = result.boxes.cls.cpu().to(torch.int64)
        image_size = result.orig_shape

        def _plot(im):
            return result.plot()

        return DetectionResult(boxes=boxes, classes=classes, image_size=image_size, _plot_impl=_plot)


def build_detector(settings: dict) -> BaseDetector:
    backend = (settings or {}).get('backend', 'detectron2').lower()
    if backend == 'detectron2':
        return Detectron2Detector(settings)
    elif backend == 'yolo':
        return YOLODetector(settings)
    else:
        raise ValueError(f"Unsupported detector backend: {backend}")
