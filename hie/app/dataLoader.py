import os
import re
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from icdmappings import Mapper
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, Dataset, dataloader,random_split
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from functools import reduce
import gc
from copy import deepcopy

pd.set_option("future.no_silent_downcasting", True)
pd.set_option("mode.chained_assignment", None)

# 1. Check default dtype
default_dtype = torch.get_default_dtype()
print(f"1. Default PyTorch dtype: {default_dtype}")
class HIEDATA2(Dataset[Any]):
    def __init__(
        self,
        dataLocation="/code/app/data/",
        #cacheRowLimit=200,
        #cacheDir="./cache",
        cacheFile="./large_cache.pkl",
        debug=False,
    ):
        super().__init__()
        self.debug = debug
        self.dataPath = Path(dataLocation)
        self.cacheFile = Path(cacheFile)

        if not self.dataPath.is_dir():
            raise FileNotFoundError("Data directory doesn't exist")
        if debug:
            print("Found data directory")
        #self.cacheRowLimit = cacheRowLimit
        self.dataFiles = list(self.dataPath.glob("*.pkl"))
        if not len(self.dataFiles):
            raise FileNotFoundError("No PKLs found in data directory")
        if debug:
            print(f"Found {len(self.dataFiles)} PKL files")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if debug:
            print(f"Using {self.device}")
        self.labelEncoder = LabelEncoder()
        #self.currentFileIdx = None

        self.IDColumn = "Patient IDX"
        self.categoricalColumns = {
            "PatientAdministrationGenderCode": None,
            "Smoking Status": None,
            # "Urine Routine",
        }
        self.categEmbedColumns = {
            "SnomedEmbed": [
                "SNOMED Codes 0",
                "SNOMED Codes 1",
                "SNOMED Codes 2",
                "SNOMED Codes 3",
                "SNOMED Codes Other",
            ],
            "ProcedureEmbed": [
                "Procedure Codes 0",
                "Procedure Codes 1",
                "Procedure Codes 2",
                "Procedure Codes 3",
                "Procedure Codes Other",
            ],
        }
        self.ignoreColumns = [
            "ICD-10 Codes 0",
            "ICD-10 Codes 1",
            "ICD-10 Codes Other",
            "ICD-10 Codes 2",
            "ICD-10 Codes 3",
        ]
        self.cliNotesColumns = [
            "Projected_Med_Embeddings",
            "Projected_Note_Embeddings",
        ]
        self.yVar = [
            "Chronic kidney disease all stages (1 through 5)",
            "Acute Myocardial Infarction",
            "Hypertension Pulmonary hypertension",
            "Ischemic Heart Disease",
        ]
        self.yVarList = {
            "Diabetes": [
                "Type 1 Diabetes",
                "Type II Diabetes",
            ]
        }
        self.continuousColumns = [
            "PatientBirthDateTime",
            "Systolic Blood Pressure",
            "Diastolic Blood Pressure",
            "Body Weight",
            "Body Height",
            "BMI",
            "Body Temperature",
            "Heart Rate",
            "Oxygen Saturation",
            "Respiratory Rate",
            "Hemoglobin A1C",
            "Blood Urea Nitrogen",
            "Bilirubin lab test",
            "Troponin Lab Test",
            "Ferritin",
            "Glucose Tolerance Testing",
            "Cerebral Spinal Fluid (CSF) Analysis",
            "Arterial Blood Gas",
            "Comprehensive Metabolic Panel",
            "Chloride  Urine",
            "Calcium in Blood  Serum or Plasma",
            "Magnesium in Blood  Serum or Plasma",
            "Magnesium in Urine",
            "Chloride  Blood  Serum  or Plasma",
            "Creatinine  Urine",
            "Creatinine  Blood  Serum  or Plasma",
            "Phosphate Blood  Serum  or Plasma",
            "Coagulation Assay",
            "Complete Blood Count",
            "Creatine Kinase Blood  Serum  or Plasma",
            "D Dimer Test",
            "Electrolytes Panel Blood  Serum  or Plasma",
            "Inflammatory Markers (CRP) Blood  Serum  or Plasma",
            "Lipid Serum  or Plasma",
            "Sputum Culture",
            "Urine Collection 24 Hours",
            # "Urine Routine", This is string, should be included in categorical
        ]
        self.contData = []
        self.categData = []
        self.clinData = []
        self.labels = []
        self.dataLen = 0
        maxRows = 0
        self.labelEncoder = LabelEncoder()
        self.clinicalbert_model_name = (
            "emilyalsentzer/Bio_ClinicalBERT"  # "medicalai/ClinicalBERT"
        )
        self.clinicalbert_model = AutoModel.from_pretrained(
            self.clinicalbert_model_name
        ).to(self.device)
        self.clinicalbert_tokenizer = AutoTokenizer.from_pretrained(
            self.clinicalbert_model_name
        )
        if debug:
            print("Pre Processing Data and creating Cache...")
        initCategData = []
        for f in tqdm(self.dataFiles):
            df = pd.read_pickle(f)
            initCategData.append(
                df[list(self.categoricalColumns.keys())].astype("string")
            )
        initCategData = pd.concat(initCategData, ignore_index=True)
        for categLabel in self.categoricalColumns:
            self.categoricalColumns[categLabel] = LabelEncoder().fit(
                initCategData[categLabel]
            )
        for f in tqdm(self.dataFiles):
            df = pd.read_pickle(f)
            patientIDXs = df[self.IDColumn].unique()
            for patientID in tqdm(patientIDXs):
                contStack = []
                categStack = []
                clinStack = []
                # print(patientID)
                patientRows = df.loc[df[self.IDColumn] == patientID]
                patientRows = patientRows.ffill().bfill().fillna(0)
                contData = patientRows[self.continuousColumns]
                contData = contData.reset_index().drop(columns="index")
                clinicalEmbeddings = patientRows[self.cliNotesColumns]
                # categData = self.labelEncoder.fit_transform(
                # patientRows[self.categoricalColumns].fillna(0)
                # )
                for categEmbed in self.categEmbedColumns:
                    subset = (
                        patientRows[self.categEmbedColumns[categEmbed]]
                        .reset_index()
                        .drop(columns="index")
                    )
                    newData = []
                    subset = subset.astype("string")
                    # print(patientID, categEmbed)
                    for i in range(len(subset)):
                        # print(i)
                        newData.append(
                            self.genEmbeddings("".join(subset.loc[i].to_list()))
                        )
                    clinicalEmbeddings[categEmbed] = newData
                    # print(clinicalEmbeddings)
                    # print("-------------------")
                    del newData
                # for i in clinicalEmbeddings.columns:
                #     print(len(clinicalEmbeddings[i].to_numpy()[0]))
                clinicalEmbeddings = clinicalEmbeddings.to_numpy()
                categData = patientRows[self.categoricalColumns.keys()].to_numpy()
                for i, d in enumerate(self.categoricalColumns):
                    categData[:, i] = self.categoricalColumns[d].transform(
                        categData[:, i]
                    )
                # for d in range(len(contData["PatientBirthDateTime"])):
                #     contData.at[d, "PatientBirthDateTime"] = self.dateToInt(
                #         contData["PatientBirthDateTime"][d]
                #     )
                for col in contData:
                    contData[col] = pd.to_numeric(contData[col])
                # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                #     print(contData)
                # Z Score Norming continuous values.
                contData = (
                    ((contData - contData.mean()) / (contData.std() + 1e-100))
                    .fillna(0)
                    .to_numpy()
                )
                assert (
                    len(contData) == len(categData) == len(clinicalEmbeddings)
                ), print(
                    f"{len(contData)}, {len(categData)}, {len(clinicalEmbeddings)}, {patientID}"
                )
                labels = patientRows[self.yVar]
                for col in self.yVarList:
                    orValues = reduce(
                        lambda a, b: a | b, patientRows[self.yVarList[col]].T.to_numpy()
                    )
                    labels.loc[:, col] = orValues
                labels = (
                    labels.reset_index()
                    .drop(columns="index")
                    .replace("", np.nan)
                    .fillna(0)
                    .astype(int)
                    .to_numpy()
                )
                for dRow in range(len(labels)):
                    contStack.append(contData[dRow])
                    categStack.append(categData[dRow])
                    clinStack.append(np.stack(clinicalEmbeddings[dRow]))
                    if np.any(labels[dRow]):
                        # print(dRow)
                        self.contData.append(np.stack(deepcopy(contStack)))
                        self.categData.append(np.stack(deepcopy(categStack)))
                        self.clinData.append(np.stack(deepcopy(clinStack)))
                        self.labels.append(np.stack(deepcopy(labels[dRow])))
                        self.dataLen += 1
                        # if len(self.contData) > self.cacheRowLimit:
                        #     self.cacheToDisk()
                    else:
                        pass
                if self.contData:
                    if maxRows < len(self.contData[-1]):
                        maxRows = len(self.contData[-1])
                del contStack, categStack, clinStack

                # break
                # break

        print(f"Max roll up Rows: {maxRows}")
        for i, d in enumerate(self.contData):
            self.contData[i] = np.vstack(
                [
                    d,
                    np.zeros(
                        (maxRows - d.shape[0], *d.shape[1:]),
                        dtype=d.dtype,
                    ),
                ]
            )
        for i, d in enumerate(self.categData):
            self.categData[i] = np.vstack(
                [
                    d,
                    np.zeros(
                        (maxRows - d.shape[0], *d.shape[1:]),
                        dtype=d.dtype,
                    ),
                ]
            )
        for i, d in enumerate(self.clinData):
            self.clinData[i] = np.vstack(
                [
                    d,
                    np.zeros(
                        (maxRows - d.shape[0], *d.shape[1:]),
                        dtype=d.dtype,
                    ),
                ]
            )
        # print(np.stack(self.contData).shape)
        self.contData = torch.tensor(self.contData)
        self.contData = self.contData.reshape((len(self.contData), -1))
        self.categData = torch.tensor(np.array(self.categData).astype(int))
        self.categData = self.categData.reshape((len(self.categData), -1))
        self.clinData = torch.tensor(self.clinData)
        self.clinData = self.clinData.reshape((len(self.clinData), -1))
        # print(self.labels)
        self.contDataInputShape = self.contData.shape[-1]
        self.categDataInputShape = self.categData.shape[-1]
        self.clinDataInputShape = self.clinData.shape[-1]
        self.labels = torch.tensor(np.array(self.labels).astype(int))
        self.labelOutputShape = self.labels.shape[-1]

        if debug:
            print("Creating single large cache file")
        self.createLargeCache()
        if debug:
            print("Loading cache into memory")
        self.loadCacheIntoMemory()
        if debug:
            print("Done initializing Dataset")

    def createLargeCache(self):
        torch.save(
            {
                "cont": self.contData,
                "categ": self.categData,
                "cli": self.clinData,
                "labels": self.labels,
            },
            self.cacheFile,
        )
        if self.debug:
            print(f"Created large cache file: {self.cacheFile}")

    def loadCacheIntoMemory(self):
        if self.debug:
            print("Loading cache into memory")
        data = torch.load(self.cacheFile)
        self.contData = data["cont"]
        self.categData = data["categ"]
        self.clinData = data["cli"]
        self.labels = data["labels"]
        self.dataLen = len(self.labels)

    def __getitem__(self, index):
        return (
            self.contData[index],
            self.categData[index],
            self.clinData[index],
            self.labels[index],
        )

    def __len__(self):
        return self.dataLen

    def getTotalRowCount(self):
        totalRows = 0
        for dataFile in self.dataFiles:
            totalRows += len(pd.read_pickle(dataFile))
        return totalRows

    def getShapes(self):
        return (
            self.contDataInputShape,
            self.categDataInputShape,
            self.clinDataInputShape,
            self.labelOutputShape,
        )

    def chunkText(self, text, chunkSize=512):
        chunks = []
        idx = 0
        while idx < len(text):
            end = min(idx + chunkSize, len(text))
            chunk = text[idx:end]
            if end < len(text) and not re.match(r"\b\w+\b$", chunk):
                chunk += " " + text[end]
                end += 1
            chunks.append(chunk)
            idx = end
        return chunks

    def genEmbeddings(self, text):
        chunks = self.chunkText(text)
        embeddings = []
        for chunk in chunks:
            encoded_input = self.clinicalbert_tokenizer(
                chunk, return_tensors="pt", padding="max_length", truncation=True
            )
            encoded_input = encoded_input.to(self.device)
            with torch.no_grad():
                output = self.clinicalbert_model(**encoded_input)
                last_hidden_state = output.last_hidden_state
                chunk_embedding = torch.mean(last_hidden_state, dim=1)
                embeddings.append(chunk_embedding.cpu().detach().numpy())
        if embeddings:
            return np.mean(np.concatenate(embeddings), axis=0)
        else:
            return np.zeros((768,))


def loadHIEDATA(
    dataDir="/code/app/data/",
    trainBatchSize=8,
    testBatchSize=4,
):
    dataset = HIEDATA2(dataLocation=dataDir, debug=True)

    # Calculate lengths for train and test
    total_length = len(dataset)
    train_length = int(0.8 * total_length)
    test_length = total_length - train_length

    # Split the dataset
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_length, test_length],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    trainLoader = DataLoader(
        train_dataset, batch_size=trainBatchSize, shuffle=False
    )  # check if shuffle should be true in seq models
    testLoader = DataLoader(test_dataset, batch_size=testBatchSize, shuffle=False)
    return trainLoader, testLoader, dataset.categoricalColumns, dataset.getShapes()