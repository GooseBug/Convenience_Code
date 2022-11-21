import numpy as np
from typing import Union

import pydicom
from pydicom import dcmread, uid
from pydicom.pixel_data_handlers import util



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
