from bef4llm.definitions import SapSamConstants

from bef4llm.datasets.sap_sam.libsam.image_generator import ImageGenerator
from bef4llm.resource_controller.file_extractor import zip_extractor
from bef4llm.datasets.model_collection import download_any_format

import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import json
import pandas as pd
import time
import io

MAX_TRACE_LENGTH = 128
SLEEP_TIME = 0.1

""" SAP Signavio Academic Models """
XML_URL = "https://zenodo.org/record/7012043/files/sap_sam_2022.zip?download=1"

SIGNAVIO = {
    "link": XML_URL,
    "model_type": "CSV",
    "destination_dir": f"{SapSamConstants.SAP_MODEL_PATH.value}",
    "filename": "sap_sam_models.zip",
    "dir": True
}


class SignavioDataset:
    """
    This class is used to load the Signavio dataset.
    """

    def __init__(self):
        """
        Constructor of the class.
        :param path_to_dataset: Path to the dataset.
        """

    def downlaod_csv(self):
        """
        downloads zip file from signavio website
        """
        if not os.path.isdir(f"{SapSamConstants.MODEL_PATH.value}/sapsam"):
            os.makedirs(SapSamConstants.MODEL_PATH.value + "/sapsam")

        self.path_to_dataset = f"{SapSamConstants.MODEL_PATH.value}/sapsam"

        if not os.path.isfile(f"{SapSamConstants.SAP_MODEL_PATH.value}/sap_sam_models.zip"):
            download_any_format(SIGNAVIO["link"], SIGNAVIO["filename"], SIGNAVIO["destination_dir"], compressed=True)
            print("download done")
        else:
            print("Signavio dataset already downloaded")


    def extract_zip(self):
        """
        extracts zip files of the file sap_sam_models.zip
        """
        if not os.path.isdir(f"{SapSamConstants.SAP_MODEL_PATH.value}/csv"):
            os.makedirs(SapSamConstants.SAP_MODEL_PATH.value + "/csv/")

        if not (os.path.isfile(f"{SapSamConstants.SAP_MODEL_PATH.value}/csv/sap_sam_2022/models/0.csv")):
            print("extract files from sap_sam_models.zip")
            zip_extractor(zip_file_path=f"{SapSamConstants.SAP_MODEL_PATH.value}/sap_sam_models.zip",
                          dest_dir=f"{SapSamConstants.SAP_MODEL_PATH.value}/csv")

        else:
            print("All files are already extracted")


    def load_dataset(self, check_relevance=True):
        """
        Loads the dataset.
        """
        self.downlaod_csv()
        self.extract_zip()
        self.translate_models()

    def translate_models(self):
        """
        converts the csv files to the fitting fileformat, whereby each model is in a seperate file
        -> ID = model_id
        """
        print("Translating models")
        csv_path = SapSamConstants.CSV_PATH.value + "/sap_sam_2022/models"
        num_csv = len(os.listdir(csv_path))
        csv_counter = 0

        # make directories
        if not os.path.isdir(SapSamConstants.BPMN_PATH.value):
            os.makedirs(SapSamConstants.SAP_MODEL_PATH.value + "/bpmn")

        print("Start translation csv files")
        for csv in tqdm(os.listdir(csv_path)):
            csv_counter += 1

            # get dataframe in csv
            df_raw = pd.read_csv(f"{csv_path}/{csv}")
            print(f"Processing {csv_counter}/{num_csv} -> {csv_counter / num_csv * 100}%")

            if not os.path.isdir(SapSamConstants.SAP_MODEL_PATH.value + "/bpmn/" + csv.split(".")[0]):
                os.makedirs(SapSamConstants.SAP_MODEL_PATH.value + "/bpmn/" + csv.split(".")[0])

            for i in range(len(df_raw)):
                model_namespace = df_raw["Namespace"][i]
                #if model == 1000:
                 #   break
                #model += 1

                # check if BPMN is already converted
                model_id = df_raw["Model ID"][i]
                if os.path.isfile(SapSamConstants.BPMN_PATH.value + f"/{csv.split(".")[0]}/{model_id}.bpmn"):
                    #print(f" Model {model_id} is already converted to an BPMN file")
                    continue

                model_json = df_raw["Model JSON"][i]

                if (model_namespace == SapSamConstants.BPMN2_NAMESPACE.value):
                    #print(model_id, "is translated into an BPMN file")
                    self.translate_BPMN(model_json=model_json, model_namespace=model_namespace,
                                        model_id=model_id, csv=csv.split(".")[0])

                #print(f"row {i}/{(len(df_raw) * num_csv)} -> {i / (len(df_raw) * num_csv)}%")

            #if csv_counter == 3:
                #break

        print("Finished translation of csv files")

    def translate_BPMN(self, model_json, model_namespace, model_id, csv):
        """
        Translates the json string of a model to bpmn file format
        """
        #elements = parser.get_elements_flat(model_json)
        gen = ImageGenerator()

        # If the name cannot be obtained, the diagram can be named manually or by using a dummy label
        try:
            model_name = json.loads(model_json)['properties']['name']
        except:
            model_name = "dummy"

        # convert to XML -> BPMN
        try:
            xml_request = io.StringIO(gen.generate_xml(model_name, model_json, model_namespace).decode("utf-8"))

            # print(f"Sleep for {SLEEP_TIME} seconds")
            time.sleep(SLEEP_TIME)

            xml_tree = ET.parse(xml_request)
            root = xml_tree.getroot()
            # modelid = uuid.uuid1()
            filename = SapSamConstants.BPMN_PATH.value + f"/{csv}/{model_id}.bpmn"
            f = open(filename, "w")
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')

            try:
                f.write(ET.tostring(root, encoding="unicode", method="xml"))
                # print(f"Successfully converted {csv_path} at row {i} with name {model_name}")
            except:
                print(f"Failed to convert because of non unicode characters")
                return

            # encoding with utf8
            try:
                with open(filename, "r") as f:
                    text = f.read()
                with open(filename, "w", encoding="utf8") as f:
                    f.write(str.encode(text).decode("utf8"))
            except:
                print(f"Failed to encode {filename}")
                return
        except Exception:
            print("BPMN can not be translated from json")
