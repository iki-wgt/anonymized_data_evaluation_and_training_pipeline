import yaml
from pycocotools.coco import COCO

class Dataset():
    """
    Representing a dataset with methods for accessing dataset information.
    Getter-only.
    """

    def __init__(self, path) -> None:
        """
        Initialize.

        Loads dataset information.
        Sets color used to visualize a bbox for some important objects.

        Arguments:
        - path (str): Path to the YAML configuration file oof the dataset.
        """

        with open(path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.coco = COCO(self.get_anno_path())
        
    def get_scene_name(self):
        """
        Get the name of the scene/dataset.

        Returns:
            str: The name of the scene.
        """

        return self.config['scene_name']

    def get_img_path(self):
        """
        Get the path to the dataset image data.

        Returns:
            str: The path to the dataset image data.
        """

        return f"{self.config['path_imgs']}"

    def get_img_path_glob(self):
        """
        Get the glob pattern for image paths.

        Returns:
            str: The glob pattern for image paths.
        """

        return f"{self.get_img_path()}/*.{self.config['image_file_format']}"

    def get_img_ids(self):
        """
        Get img ids for all images in the dataset.

        Returns:
           list: A list of all ids of the images in the dataset.
        """

        if self.config['img_ids'] is not None:
            return self.config['img_ids']
        else:
            return self.coco.getImgIds()

    def get_cat_ids(self):
        """
        Get the category IDs in the dataset.

        Returns:
            list: A list of category IDs.
        """

        if self.config['cat_ids'] is not None:
            return self.config['cat_ids']
        else:
            return self.coco.getCatIds()

    def get_anno_path(self):
        """
        Get the path to the annotation data.

        Returns:
            str: The path to the annotation data.
        """

        return f"{self.config['path_anno']}"

    def get_coco_detections_path(self, net_name):
        """
        Get the path to save detections for a specific model.

        Arguments:
        - net_name (str): Name of the model.

        Returns:
            str: The path to save detections for the model.
        """

        return f"{self.get_scene_path(net_name)}/{net_name}_detections.json"

    def get_label_save_path(self,net_name):
        """
        Get the path to save labels for a specific model.

        Arguments:
        - net_name (str): Name of the model.

        Returns:
            str: The path to save labels for the specified model.
        """

        return f"{self.get_scene_path(net_name)}/{net_name}/labels"

    def get_eval_path(self,net_name):
        """
        Get the path to save evaluation results for a specific model.

        Arguments:
        - net_name (str): Name of the model.

        Returns:
            str: The path to save evaluation results for the specified model.
        """

        return f"{self.config['path_eval']}/{self.get_scene_name()}/{net_name}_eval"

    def get_scene_path(self,net_name):
        """
        Get the path to save scene-specific data for a model.

        Arguments:
        - net_name (str): Name of the model.

        Returns:
            str: The path to save scene-specific data for the specified model.
        """
        return f"{self.config['path_detections']}/{self.get_scene_name()}"

    def get_model_path(self):
        return self.config['model_path']