import os, re
from typing import Union

import pydicom
from pydicom import dcmread, uid
from pydicom.pixel_data_handlers import util


import numpy as np
import matplotlib.pyplot as plt

import ipywidgets
from IPython.display import clear_output



def get_dicom_metaData_and_image(
    dcm_path: str
) -> Union[np.ndarray, pydicom.dataset.FileDataset]:
    """
    DICOM으로부터 필요한 데이터를 가지고 온다.
    ----------------------------------------------------------------
    이미지 데이터는 Modality LUT, VOI LUT 하여 가지고 온다.
    """
    # DICOM Object를 가지고 온다.
    dcmObject = dcmread(dcm_path)
    
    # TransferSyntaxUID가 설정되어 있다면, pixel array를 바로 가지고 온다.
    try:
        arr = dcmObject.pixel_array
    except:
        # pixel_array를 가져오기 전에 파일을 읽은 후 TransferSyntaxUID 설정
        dcmObject.file_meta.TransferSyntaxUID = uid.ImplicitVRLittleEndian  # or whatever is the correct transfer syntax for the file
        arr = dcmObject.pixel_array
        
    # Apply rescale operation (if required, otherwise returns arr unchanged)
    rescaled_arr = util.apply_modality_lut(arr, dcmObject)
    # Apply windowing operation (if required, otherwise returns arr unchanged)
    img = util.apply_voi_lut(rescaled_arr, dcmObject, index=0)
        
    return img, dcmObject




def min_max_scaling(np_array: np.ndarray) -> np.ndarray:
    """
    np array에 대하여 min-max scaling 한다.
    """
    min_score = np.min(np_array)
    max_score = np.max(np_array)
    
    return (np_array-min_score)/(max_score-min_score)


        
        
class dicom_Viewer:

    def __init__(self, dcm_directory_path: str):
        
        self.dcm_directory_path = dcm_directory_path

        
    # 순차적으로 3D 영상을 출력한다.
    def print_sequentially(self):
        """
        dcm_directory_path에 있는 모든 DICOM 파일들을 순서대로 보여준다.
        """
        dcm_file_list = self.get_dcm_file_list()
        dcm_file_list = sorted(dcm_file_list)
        
        for dcm_file in dcm_file_list:
            
            dcm_file_path = f"{self.dcm_directory_path}/{dcm_file}"
            dcm_array, _ = get_dicom_metaData_and_image(dcm_file_path)
            dcm_array = min_max_scaling(dcm_array)
            
            plt.figure(figsize=(12, 12))
            plt.imshow(dcm_array, cmap='gray')
            plt.show()
            clear_output(wait=True)
            

            
    def Slide_viewer(self, slide_start_point=0):
        """
        Slide를 이용하여 영상을 보여준다.
        ---------------------------------------------------
        slide_start_point:
        >>> 슬라이드의 첫 위치
        """
        # 3D 영상으로부터 2D 슬라이드를 하나 보여준다.
        def cut_viewer_from_3D_image(idx: int):

            cut_img = img_Tensor[idx, :, :]

            plt.figure(figsize=(12, 12))
            plt.imshow(cut_img, cmap="gray")
            plt.title(f"Slide Number: {idx}", fontsize = 25, pad = 20)

            plt.show()

        img_Tensor = self.import_3D_image()
        Depth = img_Tensor.shape[0] - 1

        ipywidgets.interact(
            cut_viewer_from_3D_image,
            idx=ipywidgets.IntSlider(
                value=slide_start_point,
                min=0,
                max=Depth,
                layout=ipywidgets.Layout(width='800px', description='Blue handle')
            )
        )

            
            
    # 미리 2D DICOM 파일들을 가지고 온다.
    def import_3D_image(self) -> np.ndarray:

        dcm_file_list = self.get_dcm_file_list()
        dcm_file_list = sorted(dcm_file_list)

        stack_list = []

        for dcm_file in dcm_file_list:

            dcm_file_path = f"{self.dcm_directory_path}/{dcm_file}"
            dcm_array, _ = get_dicom_metaData_and_image(dcm_file_path)
            dcm_array = min_max_scaling(dcm_array)
            stack_list.append(dcm_array)

        return np.array(stack_list)
    


    # DICOM 파일만 가지고 온다.
    def get_dcm_file_list(self) -> list:

        # file의 목록들을 다 가지고 온다.
        file_list = os.listdir(self.dcm_directory_path)

        dcm_list = []

        for oneFileName in file_list:
            end_is_point_dcm = len(re.findall("[.]dcm$", oneFileName))

            if end_is_point_dcm == 1:
                dcm_list.append(oneFileName)

        return dcm_list
