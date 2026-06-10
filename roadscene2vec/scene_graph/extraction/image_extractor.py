import os
import pdb
from pathlib import Path

import cv2
from os.path import isfile, join
import roadscene2vec.data.dataset as ds
from roadscene2vec.scene_graph.extraction.extractor import Extractor as ex
from roadscene2vec.scene_graph.scene_graph import SceneGraph
from roadscene2vec.scene_graph.extraction.bev import bev
from tqdm import tqdm

'''RealExtractor initializes relational settings and creates an ImageSceneGraphSequenceGenerator object to extract scene graphs using raw image data.'''
class RealExtractor(ex):
    def __init__(self, config):
        super(RealExtractor, self).__init__(config) 

        self.input_path = self.conf.location_data['input_path']
        self.dataset = ds.SceneGraphDataset(self.conf)
        self.object_detection_settings = getattr(self.conf, "object_detection_settings", {}) or {}
        self.detector_backend = self.object_detection_settings.get("backend", "detectron2").lower()

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(self.input_path)

        # build a detector adapter via the unified factory
        from roadscene2vec.scene_graph.extraction.detectors import build_detector

        detector_settings = dict(self.object_detection_settings) if self.object_detection_settings is not None else {}
        detector_settings.setdefault('backend', self.detector_backend)
        self.detector = build_detector(detector_settings)
        self.coco_class_names = self.detector.names

        # bev setup
        self.bev = bev.BEV(config.image_settings['BEV_PATH'], mode='deploy')


    '''Load scenegraphs using raw image frame tensors'''
    def load(self): #seq_tensors[seq][frame/jpgname] = frame tensor
        try:
            all_sequence_dirs = [x for x in Path(self.input_path).iterdir() if x.is_dir()]
            all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0]))  
            self.dataset.folder_names = [path.stem for path in all_sequence_dirs]
            for path in tqdm(all_sequence_dirs):
                seq = int(path.stem.split('_')[0])
                label_path = (path/"label.txt").resolve()
                ignore_path = (path/"ignore.txt").resolve()
                if ignore_path.exists(): #record ignored sequences, and only load the sequences that were not ignored
                    with open(str(path/"ignore.txt"), 'r') as label_f:
                        ignore_label = int(label_f.read())
                        if ignore_label:
                            self.dataset.ignore.append(seq)
                            continue #skip to next seq if ignore path exists
                seq_images = self.load_images(path)
            
                self.dataset.scene_graphs[seq] = {}
                for frame, img in seq_images.items():
                    out_img_path = None
                    bounding_boxes = self.get_bounding_boxes(img_tensor=img, out_img_path=out_img_path)
                    
                    scenegraph = SceneGraph(self.relation_extractor,    
                                                bounding_boxes = bounding_boxes, 
                                                bev = self.bev,
                                                coco_class_names=self.coco_class_names, 
                                                platform=self.dataset_type)

                    self.dataset.scene_graphs[seq][frame] = scenegraph
                self.dataset.action_types[seq] = "lanechange" 
                if label_path.exists():
                    with open(str(path/'label.txt'), 'r') as label_file:
                        lines = label_file.readlines()
                        l0 = 1.0 if float(lines[0].strip().split(",")[0]) >= 0 else 0.0 
                        self.dataset.labels[seq] = l0

        except Exception as e:
            pdb.set_trace()
            import traceback
            print('We have problem creating the real image scenegraphs')
            print(e)
            traceback.print_exc()
    
    #returns a numpy array representation of a sequence of images in format (H,W,C)
    def load_images(self, path):
        raw_images_loc = (path/'raw_images').resolve()
        images = sorted([Path(f) for f in os.listdir(raw_images_loc) if isfile(join(raw_images_loc, f)) and ".DS_Store" not in f and "Thumbs" not in f], key = lambda x: int(x.stem.split(".")[0]))
        images = [join(raw_images_loc,i) for i in images] 
        sequence_tensor = {}
        modulo = 0
        acc_number = 0
        if(self.framenum != None):
            modulo = int(len(images) / self.framenum)  #subsample to frame limit
        if(self.framenum == None or modulo == 0):
            modulo = 1
        for i in range(0, len(images)):
            if (i % modulo == 0 and self.framenum == None) or (i % modulo == 0 and acc_number < self.framenum):
                image_path = images[i]
                frame_num = int(Path(image_path).stem)
                im = cv2.imread(str(image_path), cv2.IMREAD_COLOR) 
                sequence_tensor[frame_num] = im 
                acc_number += 1
        return sequence_tensor
        
    def get_bounding_box_annotated_image(self, im):
        try:
            # Detector implementations provide a `plot` implementation via DetectionResult
            results = self.detector.predict(im, conf=float(self.object_detection_settings.get("confidence", 0.5)),
                                            iou=float(self.object_detection_settings.get("iou", 0.7)),
                                            device=self.object_detection_settings.get("device", None))
            return results.plot(im)
        except Exception:
            # Fallback: try calling detector-specific plot if available
            try:
                return self.detector.model.plot(im)  # type: ignore[attr-defined]
            except Exception:
                raise
            
    def get_bounding_boxes(self, img_tensor, out_img_path=None):
        im = img_tensor
        # Use detector adapter to get a unified DetectionResult
        res = self.detector.predict(
            im,
            conf=float(self.object_detection_settings.get("confidence", 0.5)),
            iou=float(self.object_detection_settings.get("iou", 0.7)),
            device=self.object_detection_settings.get("device", None),
        )
        boxes = res.boxes
        classes = res.classes
        image_size = res.image_size

        if out_img_path:
            # We can use `Visualizer` to draw the predictions on the image.
            out = self.get_bounding_box_annotated_image(im)
            cv2.imwrite(out_img_path, out)

        # todo: after done scp to server
        # crop im to remove ego car's hood
        # find threshold then remove from pred_boxes, pred_classes, check image_size
        bounding_boxes = boxes, classes, image_size
        return bounding_boxes

    
    '''Returns SceneGraphDataset object containing scengraphs, labels, and action types'''
    def getDataSet(self):
        try:
            return self.dataset
        except Exception as e:
            import traceback
            print('We have problem creating scenegraph dataset object from the extracted real image scenegraphs')
            print(e)
            traceback.print_exc()
    
   
    
            
