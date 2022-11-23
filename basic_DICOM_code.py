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
        self.img_arr = None
        self.header_dict = None
        self.only_slide_number = None

        
        
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
            
            plt.figure(figsize=(10, 10))
            plt.imshow(dcm_array, cmap='gray')
            plt.show()
            clear_output(wait=True)
            

            
    def Slide_viewer(self, slide_start_point: int = 0, only_slide_number: bool = True):
        """
        Slide를 이용하여 영상을 보여준다.
        ---------------------------------------------------
        slide_start_point:
        >>> 슬라이드를 보여줄 첫 위치
        
        only_slide_number:
        >>> title에서 슬라이드의 번호만 보여줄 것인지 여부
        """
        self.img_arr, self.header_dict = self.extract_need_dicom_information()
        Depth = self.img_arr.shape[0] - 1
        
        # Slide 번호만 출력할 것인지
        self.only_slide_number = only_slide_number
        
        # Widget을 생성한다.
        ipywidgets.interact(
            self.cut_viewer_from_3D_image,
            idx=ipywidgets.IntSlider(
                value=slide_start_point,
                min=0,
                max=Depth,
                layout=ipywidgets.Layout(
                    width='800px', description='Blue handle'
                )
            )
        )
        

        
    # 3D 영상으로부터 2D 슬라이드를 하나 보여준다.
    def cut_viewer_from_3D_image(self, idx: int):

        cut_img = self.img_arr[idx, :, :]
        
        if self.only_slide_number:
            title = f"Slide Number: {idx}"
        else:
            title = self.make_DICOM_header_info_title(idx)

        plt.figure(figsize=(10, 10))
        plt.imshow(cut_img, cmap="gray")
        plt.title(title, fontsize = 20, pad = 20)

        plt.show()
        
        
        
    def make_DICOM_header_info_title(self, idx):
        
        result = f"""
        Slide Number: {idx}
        InstanceNumber: {self.header_dict[idx]["InstanceNumber"]}
        ImagePositionPatient: {self.header_dict[idx]["ImagePositionPatient"]}
        SliceLocation: {self.header_dict[idx]["SliceLocation"]}
        SeriesInstanceUID: {self.header_dict[idx]["SeriesInstanceUID"]}
        SeriesNumber: {self.header_dict[idx]["SeriesNumber"]}
        ImageOrientationPatient: {self.header_dict[idx]["ImageOrientationPatient"]}
        """
        return result
        
        

    # 미리 2D DICOM 파일들을 가지고 온다.
    def extract_need_dicom_information(self):

        dcm_file_list = self.get_dcm_file_list()
        dcm_file_list = sorted(dcm_file_list)

        img_stack_list = []
        header_dict = dict()

        for i, dcm_file in enumerate(dcm_file_list):

            dcm_file_path = f"{self.dcm_directory_path}/{dcm_file}"

            # DICOM 정보를 가지고 온다.
            dcmImage, headerInfo_dict = extract_dicom_information(dcmPath=dcm_file_path).process()
            img_stack_list.append(dcmImage)
            header_dict[i] = headerInfo_dict
            
        img_arr = np.array(img_stack_list)
            
        return img_arr, header_dict
    


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
    
    
    
    
class extract_dicom_information:
    
    
    def __init__(self, dcmPath):
        
        self.dcmPath = dcmPath
        self.dcm_object = None
        self.warning_phrase = "This DICOM has no target metadata."
        
        
    def process(self):
        
        dcmImage, self.dcm_object = get_dicom_metaData_and_image(self.dcmPath)
        
        headerInfo_dict = {
            "InstanceNumber":self.get_InstanceNumber(),
            "ImagePositionPatient":self.get_ImagePositionPatient(),
            "SliceLocation":self.get_SliceLocation(),
            "SeriesInstanceUID":self.get_SeriesInstanceUID(),
            "SeriesNumber":self.get_SeriesNumber(),
            "ImageOrientationPatient":self.get_ImageOrientationPatient()
        }
        
        return dcmImage, headerInfo_dict
        
        
    def get_InstanceNumber(self):
        try:
            result = self.dcm_object.InstanceNumber
        except:
            result = self.warning_phrase
        return result


    def get_ImagePositionPatient(self):
        try:
            result = self.dcm_object.ImagePositionPatient
        except:
            result = self.warning_phrase
        return result


    def get_SliceLocation(self):
        try:
            result = self.dcm_object.SliceLocation
        except:
            result = self.warning_phrase
        return result


    def get_SeriesInstanceUID(self):
        try:
            result = self.dcm_object.SeriesInstanceUID
        except:
            result = self.warning_phrase
        return result
    
    
    def get_SeriesNumber(self):
        try:
            result = self.dcm_object.SeriesNumber
        except:
            result = self.warning_phrase
        return result
    
    
    def get_ImageOrientationPatient(self):
        try:
            result = self.dcm_object.ImageOrientationPatient
        except:
            result = self.warning_phrase
        return result
