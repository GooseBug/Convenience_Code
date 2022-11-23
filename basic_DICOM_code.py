import os, re
from typing import Union

import pydicom
from pydicom import dcmread, uid
from pydicom.pixel_data_handlers import util


import numpy as np
import matplotlib.pyplot as plt

import ipywidgets
from IPython.display import clear_output



class dicom_Viewer:

    def __init__(self, dcm_directory_path: str):
        """
        설명
        ----------------------------------------------------------------
        dicom_Viewer는 DICOM이 들어 있는 디렉터리에 대하여,
        DICOM을 순서대로 출력한다.
        ----------------------------------------------------------------
        
        
        주요 Method
        ----------------------------------------------------------------
        1) print_sequentially
        >>> 3D 이미지를 File의 이름 순으로 천천히 출력한다.
        2) print_one_index_slide
        >>> 슬라이드 순서를 고려하여, 원하는 슬라이드 한 장을 출력한다.
        3) Slide_viewer
        >>> 3D 이미지를 Slide를 움직여 원하는 형태로 출력한다.
        ----------------------------------------------------------------
        """
        self.dcm_directory_path = dcm_directory_path
        self.img_arr = None
        self.header_dict = None
        self.only_slide_number = None
        self.title_color = None

        
        
    # 순차적으로 3D 영상을 출력한다.
    def print_sequentially(self, title_color="white"):
        """
        dcm_directory_path에 있는 모든 DICOM 파일들을 순서대로 보여준다.
        """
        dcm_file_list = self.get_dcm_file_list()
        dcm_file_list = sorted(dcm_file_list)
        total_slide_num = len(dcm_file_list)
        
        for slide_num, dcm_file in enumerate(dcm_file_list):
            
            dcm_file_path = f"{self.dcm_directory_path}/{dcm_file}"
            dcm_array, _ = get_dicom_metaData_and_image(dcm_file_path)
            dcm_array = min_max_scaling(dcm_array)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(dcm_array, cmap='gray')
            
            title = f"Slide number: {slide_num+1}/{total_slide_num}"
            self.title_color = title_color
            plt.title(title, fontsize = 20, pad = 20, color=self.title_color)
            
            plt.show()
            clear_output(wait=True)
            
            
            
    # 3D 영상 중 특정 index에 대한 Slide 하나를 출력한다.
    def print_one_index_slide(self, idx, dicom_header_title: bool = False, title_color = "white"):
        """
        dcm_directory_path에 있는 모든 DICOM 파일 중 idx에 해당하는 이미지 하나를 보여준다.
        """
        # directory는 1부터 시작이나, Python은 0부터 시작이므로 시작 위치를 맞춘다.
        python_idx = idx-1
        
        # 대상 슬라이드의 경로 정보를 가지고 온다. 
        dcm_file_list = self.get_dcm_file_list()
        dcm_file_list = sorted(dcm_file_list)
        dcm_file = dcm_file_list[python_idx]
        dcm_file_path = f"{self.dcm_directory_path}/{dcm_file}"
        
        # DICOM에 대한 필요 정보를 가지고 온다.
        dcmImage, headerInfo_dict = extract_dicom_information(
            dcmPath=dcm_file_path
        ).process()
        
        # title 관련 정보 정의
        self.title_color = title_color
        if self.only_slide_number:
            title = f"Slide Number: {python_idx + 1}"   # 0으로 시작하는 python의 index를 1로 시작하는 directory index로 맞춰준다.
        else:
            title = self.make_DICOM_header_info_title(
                one_header_dict = headerInfo_dict,
                idx = python_idx  # 디렉터리 내 슬라이드의 순서를 표기한다.
            )
        
        plt.figure(figsize=(10, 10))
        plt.imshow(dcmImage, cmap='gray')
        plt.title(title, fontsize = 20, pad = 20, color=self.title_color)
        plt.show()
            
            

            
    def Slide_viewer(
        self,
        slide_start_point: int = 0,
        only_slide_number: bool = True,
        title_color = "white"
    ):
        """
        Slide를 이용하여 영상을 보여준다.
        ---------------------------------------------------
        slide_start_point:
        >>> 슬라이드를 보여줄 첫 위치
        
        only_slide_number:
        >>> title에서 슬라이드의 번호만 보여줄 것인지 여부
        
        title_color:
        >>> Slide에서 title의 색깔
        """
        self.img_arr, self.header_dict = self.extract_need_dicom_information()
        Depth = self.img_arr.shape[0] - 1
        
        # title의 색을 정의한다.
        self.title_color = title_color
        
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
            title = self.make_DICOM_header_info_title(
                one_header_dict = self.header_dict[idx],
                idx = idx
            )

        plt.figure(figsize=(10, 10))
        plt.imshow(cut_img, cmap="gray")
        plt.title(title, fontsize = 20, pad = 20, color=self.title_color)

        plt.show()
        
        
        
    def make_DICOM_header_info_title(self, one_header_dict: dict, idx: str):
        
        result = f"""
        Slide Number: {idx + 1}
        InstanceNumber: {one_header_dict["InstanceNumber"]}
        ImagePositionPatient: {one_header_dict["ImagePositionPatient"]}
        SliceLocation: {one_header_dict["SliceLocation"]}
        SeriesInstanceUID: {one_header_dict["SeriesInstanceUID"]}
        SeriesNumber: {one_header_dict["SeriesNumber"]}
        ImageOrientationPatient: {one_header_dict["ImageOrientationPatient"]}
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
