import numpy as np
import os
import rasterio
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def load_raster(path, if_img=1, crop=None):
  with rasterio.open(path) as src:
    img = src.read(out_dtype=np.float32)
    if if_img==1:
      bands=[0,1,2,3]
      img = img[bands,:,:]
    if crop:
      img = img[:, -crop[0]:, -crop[1]:]
  return img


class HorizontalCycleTransform(nn.Module):
    def forward(self, img):
        img2 = torch.cat([img, img], -1)
        start = torch.randint(img.shape[-1], (1,))[0]
        new_img = img2[:,:,start:start+img.shape[-1]]
        return new_img


class HorizontalPermuteTransform(nn.Module):
    def forward(self, img):
        new_img = img[:,:,torch.randperm(img.shape[-1])]
        return new_img


class TestDataset(Dataset):
    def __init__(self, dir_sentinel, dir_landsat, dir_bioclim, metadata, cov_columns, subset, num_classes=None, transform_sentinel=None, transform_landsat=None, landsat_year_len=18,
                image_mean=False, sentinel_mask_channel=True):
        # transform_landsat corresponds to landsat and bioclim cubes combined
        self.subset = subset
        self.transform_sentinel = transform_sentinel
        self.transform_landsat = transform_landsat
        self.dir_sentinel = dir_sentinel
        self.dir_landsat = dir_landsat
        self.dir_bioclim = dir_bioclim
        self.metadata = metadata
        self.cov_columns = cov_columns
        self.num_classes = num_classes
        self.landsat_year_len = landsat_year_len
        self.image_mean = image_mean
        self.sentinel_mask_channel = sentinel_mask_channel
        if self.subset == "test" or self.subset == "po":
            self.landsat_file_sep = "_"
        elif self.subset == "train":
            self.landsat_file_sep = "-"

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        bioclim_month_len = self.landsat_year_len * 12 - 1
        survey_id = self.metadata.surveyId[idx]
        dir1, dir2 = str(survey_id)[-2:], str(survey_id)[-4:-2]
        cov = torch.tensor(self.metadata.loc[idx, self.cov_columns].values.astype(np.float32))
        lonlat = torch.tensor(self.metadata.loc[idx, ["lon","lat"]].values.astype(np.float32))
        if self.subset == "train" or self.subset == "test":
            path_landsat = os.path.join(self.dir_landsat, f"GLC25-PA-{self.subset}-landsat{self.landsat_file_sep}time{self.landsat_file_sep}series_{survey_id}_cube.pt")
        else:
            path_landsat = os.path.join(self.dir_landsat, dir1, dir2, f"GLC25-PO-train-landsat_time_series_{survey_id}_cube.pt")
        sample_landsat = torch.nan_to_num(torch.load(path_landsat, weights_only=True)[:,:,:self.landsat_year_len])
        if self.subset == "train" or self.subset == "test":
            path_bioclim = os.path.join(self.dir_bioclim, f"GLC25-PA-{self.subset}-bioclimatic_monthly_{survey_id}_cube.pt")
        else:
            path_bioclim = os.path.join(self.dir_bioclim, dir1, dir2, f"GLC25-P0-train-bioclimatic_monthly_{survey_id}_cube.pt")
        sample_bioclim = torch.nan_to_num(torch.load(path_bioclim, weights_only=True))
        tmp1 = torch.reshape(sample_bioclim, [4,-1])
        tmp2 = torch.reshape(torch.cat([tmp1[:,:1], tmp1[:,:bioclim_month_len]], axis=-1), [4,self.landsat_year_len,4,3])
        sample_bioclim_new = torch.permute(torch.mean(tmp2, -1), [0,2,1])[:2] 
        sample_landsat = torch.cat([sample_landsat, sample_bioclim_new], 0)
        if self.transform_landsat:
            sample_landsat = self.transform_landsat(sample_landsat)
        if self.image_mean:
            sample_landsat = torch.mean(sample_landsat, [-1])
        path_sentinel = os.path.join(self.dir_sentinel, dir1, dir2, f"{survey_id}.tiff")
        sample_sentinel = torch.nan_to_num(torch.from_numpy(load_raster(path_sentinel)).to(torch.float32))
        if self.transform_sentinel:
            sample_sentinel = self.transform_sentinel(sample_sentinel)
        if self.image_mean:
            sample_sentinel = torch.mean(sample_sentinel, [-1, -2])
        if self.sentinel_mask_channel:
            EPS = 1e-9
            mask = (torch.sum(torch.abs(sample_sentinel) < EPS, [0]) < EPS).to(torch.float32)
            sample_sentinel = torch.concat([sample_sentinel, mask[None,:,:]], 0)
        return sample_sentinel, sample_landsat, cov, lonlat, survey_id


class TrainDataset(TestDataset):
  def __init__(self, dir_sentinel, dir_landsat, dir_bioclim, metadata, cov_columns, label_dict, subset, num_classes, transform_sentinel=None, transform_landsat=None,
               landsat_year_len=18, image_mean=False, sentinel_mask_channel=True):
    super(TrainDataset, self).__init__(dir_sentinel, dir_landsat, dir_bioclim, metadata, cov_columns, subset, num_classes, transform_sentinel, transform_landsat,
                                       landsat_year_len, image_mean, sentinel_mask_channel)
    self.label_dict = label_dict

  def __getitem__(self, idx):
    sample_sentinel, sample_landsat, cov, lonlat, survey_id = super(TrainDataset, self).__getitem__(idx)
    species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
    label = torch.zeros(self.num_classes).scatter(0, torch.tensor(species_ids), torch.ones(len(species_ids)))
    return sample_sentinel, sample_landsat, cov, lonlat, label, survey_id


def read_train_data(path_data, cov_flag_list, sel_countries, pa_presence_threshold=1, landcover_col_ind=[0,2,3,5,8,11,12]):
    train_metadata = pd.read_csv(os.path.join(path_data, "GLC25_PA_metadata_train.csv"))
    train_metadata = train_metadata.dropna(subset="speciesId").reset_index(drop=True)
    train_metadata['speciesId'] = train_metadata['speciesId'].astype(int)
    train_metadata["speciesIdOrig"] = train_metadata['speciesId']
    tmp = train_metadata["speciesId"].value_counts() >= pa_presence_threshold
    train_metadata.loc[~train_metadata["speciesId"].isin(tmp[tmp].index), "speciesId"] = -1
    sp_categorical = train_metadata["speciesId"].astype("category").values
    num_classes = len(sp_categorical.categories)
    train_metadata['speciesId'] = sp_categorical.codes
    
    tmp = train_metadata.groupby("surveyId").agg({"surveyId":"first", "lat":"first", "lon":"first", "areaInM2":lambda x: list(x.unique()), "region":"first", "country":"first", "speciesId":list})
    train_label_series = tmp.set_index("surveyId").speciesId
    train_metadata = tmp.drop(columns=["speciesId"]).set_index("surveyId", drop=False)
    train_metadata["area"] = train_metadata["areaInM2"].apply(lambda x: 1.0 if np.isinf(x).all() else np.mean(x, where=~np.isinf(x)))
    train_metadata["areaLog"] = np.log10(train_metadata["area"])
    
    train_metadata['area'] = train_metadata['area'].fillna(train_metadata['area'].mean())
    train_metadata['areaLog'] = train_metadata['areaLog'].fillna(train_metadata['areaLog'].mean())
    country_columns = ["con"+country[:3] for country in sel_countries] + ["conOther"]
    for country, col in zip(sel_countries, country_columns[:-1]):
        train_metadata[col] = train_metadata["country"] == country
    train_metadata[country_columns[-1]] = ~train_metadata["country"].isin(sel_countries)
    train_elevation = pd.read_csv(os.path.join(path_data, "EnvironmentalValues", "Elevation", "GLC25-PA-train-elevation.csv"), index_col=0)
    train_elevation['Elevation'] = train_elevation['Elevation'].fillna((train_elevation['Elevation'].mean()))
    train_soil = pd.read_csv(os.path.join(path_data, "EnvironmentalValues", "SoilGrids", "GLC25-PA-train-soilgrids.csv"), index_col=0)
    for column in train_soil.columns: train_soil[column] = train_soil[column].fillna((train_soil[column].mean()))
    train_worldcover = pd.read_csv(os.path.join(path_data, "worldcover", "s2_pa_train_survey_points_with_worldcover.csv"), index_col=0)
    train_wcdummy = pd.get_dummies(train_worldcover["class"], prefix="wc")
    train_wcdummy.drop(columns="wc_70", inplace=True)
    train_wcdummy.drop(columns="wc_100", inplace=True)
    train_landcover = pd.read_csv(os.path.join(path_data, "EnvironmentalValues", "LandCover", "GLC25-PA-train-landcover.csv"), index_col=0)
    train_landcover = train_landcover.iloc[:, landcover_col_ind]
    train_snow = pd.read_csv(os.path.join(path_data, "EnvironmentalValues", "chelsa_snow", "pa_train_snowcover_chelsa_scd.csv"), index_col=0).sort_index()
    
    cov_name_list = [["areaLog"], ["Elevation"], country_columns, list(train_soil.columns), list(train_wcdummy.columns), list(train_landcover.columns), list(train_snow.columns)]
    cov_columns = sum([flag*name for flag, name in zip(cov_flag_list, cov_name_list)], [])
    print(cov_columns)
    train_df_list = [train_metadata, train_elevation, train_soil, train_wcdummy, train_landcover, train_snow]
    print("All rows match: ", [(train_df_list[0].index==df.index).all() for df in train_df_list[1:]])
    train_combined = pd.concat(train_df_list, axis=1)
    cov_norm_coef = train_combined.loc[:,cov_columns].agg(['mean', 'std'])
    dummy_columns = country_columns + list(train_wcdummy.columns)
    cov_norm_coef.loc["mean",dummy_columns] = 0
    cov_norm_coef.loc["std",dummy_columns] = 1
    train_combined.loc[:,cov_columns] = (train_combined.loc[:,cov_columns] - cov_norm_coef.loc["mean"]) / cov_norm_coef.loc["std"]
    return train_combined, train_label_series, sp_categorical.categories.values, cov_columns, cov_norm_coef, num_classes


def read_test_data(path_data, cov_columns, cov_norm_coef, sel_countries, landcover_col_ind=[0,2,3,5,8,11,12]):
    test_metadata = pd.read_csv(os.path.join(path_data, "GLC25_PA_metadata_test.csv")).set_index("surveyId", drop=False).sort_index()
    test_metadata.rename(columns={"areaInM2": "area"}, inplace=True)
    test_metadata.replace({"area": [np.inf, -np.inf]}, 1.0, inplace=True)
    test_metadata['areaLog'] = np.log10(test_metadata['area'])
    test_metadata['area'] = test_metadata['area'].fillna(test_metadata['area'].mean())
    test_metadata['areaLog'] = test_metadata['areaLog'].fillna(test_metadata['areaLog'].mean())
    country_columns = ["con"+country[:3] for country in sel_countries] + ["conOther"]
    for country, col in zip(sel_countries, country_columns[:-1]):
        test_metadata[col] = test_metadata["country"] == country
    test_metadata[country_columns[-1]] = ~test_metadata["country"].isin(sel_countries)
    test_elevation = pd.read_csv(os.path.join(path_data, "EnvironmentalValues", "Elevation", "GLC25-PA-test-elevation.csv"), index_col=0).sort_index()
    test_elevation = test_elevation.loc[test_elevation.index.isin(test_metadata.index)]
    test_elevation['Elevation'] = test_elevation['Elevation'].fillna((test_elevation['Elevation'].mean()))
    test_soil = pd.read_csv(os.path.join(path_data, "EnvironmentalValues", "SoilGrids", "GLC25-PA-test-soilgrids.csv"), index_col=0).sort_index()
    test_soil = test_soil.loc[test_soil.index.isin(test_metadata.index)]
    for column in test_soil.columns: test_soil[column] = test_soil[column].fillna((test_soil[column].mean()))
    test_worldcover = pd.read_csv(os.path.join(path_data, "worldcover", "pa_test_survey_points_with_worldcover.csv"), index_col=0).sort_index()
    test_wcdummy = pd.get_dummies(test_worldcover["class"], prefix="wc")
    test_wcdummy.drop(columns="wc_100", inplace=True)
    # test_wcdummy.insert(6, "wc_70", False)
    test_landcover = pd.read_csv(os.path.join(path_data, "EnvironmentalValues", "LandCover", "GLC25-PA-test-landcover.csv"), index_col=1).sort_index()
    if any(test_landcover.columns == "Unnamed: 0"):
        test_landcover.drop("Unnamed: 0", axis=1, inplace=True)
    test_landcover = test_landcover.loc[test_landcover.index.isin(test_metadata.index)]
    test_landcover = test_landcover.iloc[:, landcover_col_ind]
    test_snow = pd.read_csv(os.path.join(path_data, "EnvironmentalValues", "chelsa_snow", "pa_test_snowcover_chelsa_scd.csv"), index_col=0).sort_index()
    test_snow = test_snow.loc[test_snow.index.isin(test_metadata.index)]
    
    test_df_list = [test_metadata, test_elevation, test_soil, test_wcdummy, test_landcover, test_snow]
    print("All rows match: ", [(test_df_list[0].index==df.index).all() for df in test_df_list[1:]])
    test_combined = pd.concat(test_df_list, axis=1)
    test_combined.loc[:,cov_columns] = (test_combined.loc[:,cov_columns] - cov_norm_coef.loc["mean"]) / cov_norm_coef.loc["std"]
    return test_combined