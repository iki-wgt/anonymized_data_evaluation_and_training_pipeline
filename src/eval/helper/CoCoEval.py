"""
COCO evaluation module for object detection models. This script evaluates 
detection results using the COCO evaluation API, including standard metrics 
(mAP, precision, recall) and enhanced F-beta scores.
It saves evaluation reports and generates precision-recall curves.
"""
import sys
from io import StringIO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from helper.utils import check_path
from helper.Dataset import Dataset

class CocoEvaluation():
    """
    CocoEvaluation class is used for evaluating object detection performance using COCO metrics.
    Use case: single sub-dataset
    """

    def __init__(self, config_path, net_name) -> None:
        """
        Initialize.
        Loads dataset information, ground truth annotations and detection results.

        Arguments:
            config_path (str): Path to the configuration file.
            net_name (str): Name of the neural network model.
        """

        self.net_name = net_name
        self.data = Dataset(config_path)

        # load GT data & detection results
        print(f"Loading GT from: {self.data.get_anno_path()}")
        self.coco_GT = COCO(f'{self.data.get_anno_path()}')
        
        label_path = self.data.get_coco_detections_path(net_name)
        print(f"Loading own Detections from: {label_path}")
        self.results = self.coco_GT.loadRes(label_path)

    def eval(self):
        """
        Perform evaluation on the provided dataset and compute COCO metrics.
        Saves output of COCO API (sadly just terminal output) to file.
        Saves plots of Fscores and PR curves (but not nice...) Matlab script does better plots.
        """

        # prepare
        anno_type = 'bbox'
        coco_eval = COCOeval(self.coco_GT, self.results, anno_type)
        coco_eval.params.catIds = self.data.get_cat_ids()
        coco_eval.params.imgIds = sorted(self.data.get_img_ids())
        coco_eval.evaluate()

        coco_eval.accumulate()
        coco_eval.summarize()
        
        print(coco_eval.evalImgs[0])

        # start to capture output
        original_stdout = sys.stdout
        output_buffer = StringIO()
        sys.stdout = output_buffer

        # enhanced coco
        print("\n ### enhanced coco ###\n")
        coco_eval.accumulateFBeta()
        fscore, conf, precision, recall = coco_eval.getBestFBeta(beta=1, iouThr=0.5, classIdx=None, average='macro')
        coco_eval.summarizeFBetaScores(average='macro')
        coco_eval.printReport(beta=1, iouThr=0.5)
        coco_eval.printReport(beta=2, iouThr=0.5)
        
        # original coco style
        print("\n ### original coco ###\n")
        coco_eval.accumulate()
        coco_eval.summarize()

        # end captue output
        sys.stdout = original_stdout # Restore the original stdout
        captured_output = output_buffer.getvalue()

        # save results
        path = self.data.get_eval_path(self.net_name)
        if check_path(path):
            coco_eval.plotCocoPRCurve(f'{path}/PRCurve.jpg', classIdx=None)
            coco_eval.plotFBetaCurve(f'{path}/Fscores.jpg', betas=[1,2], iouThr=0.5, average='macro')
            with open(f'{path}/eval.txt', 'w') as file:
                file.write(captured_output)
        else:
            print("Unable to save evaluation results!")
