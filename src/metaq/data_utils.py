from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset


class MetaQDataset(Dataset):
    def __init__(self, x_list, sf_list, raw_list):
        super().__init__()
        self.x_list = x_list
        self.sf_list = sf_list
        self.raw_list = raw_list

        self.cell_num = self.x_list[0].shape[0]
        self.omics_num = len(self.x_list)

        for i in range(self.omics_num):
            self.x_list[i] = torch.from_numpy(self.x_list[i]).float()
            self.sf_list[i] = torch.from_numpy(self.sf_list[i]).float()
            self.raw_list[i] = torch.from_numpy(self.raw_list[i]).float()

    def __len__(self):
        return int(self.cell_num)

    def __getitem__(self, index):
        x_list = []
        sf_list = []
        raw_list = []
        for i in range(self.omics_num):
            x_list.append(self.x_list[i][index])
            sf_list.append(self.sf_list[i][index])
            raw_list.append(self.raw_list[i][index])
        data = {"x": x_list, "sf": sf_list, "raw": raw_list}
        return data


def preprocess(adata, data_type):
    if isinstance(adata.X, sparse.csr_matrix) or isinstance(adata.X, sparse.csc_matrix):
        adata.X = adata.X.toarray()
    raw = adata.X.copy()

    if data_type == "RNA":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
        sc.pp.log1p(adata)
        adata_ = adata.copy()
        if adata.shape[1] < 5000:
            sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        else:
            sc.pp.highly_variable_genes(adata)
        hvg_index = adata.var["highly_variable"].values
        raw = raw[:, hvg_index]
        adata = adata[:, hvg_index]
    elif data_type == "ADT":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
        sc.pp.log1p(adata)
        adata_ = adata.copy()
    elif data_type == "ATAC":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
        sc.pp.log1p(adata)
        adata_ = adata.copy()
        sc.pp.highly_variable_genes(adata, n_top_genes=30000)
        hvg_index = adata.var["highly_variable"].values
        raw = raw[:, hvg_index]
        adata = adata[:, hvg_index]

    sc.pp.scale(adata, max_value=10)
    x = adata.X

    return x, sf, raw, adata_


def load_data(
    data_path: list[str],
    data_type: list[str],
    metacell_num: Optional[int] = None,
    metacell_frac: Optional[float] = None,
    batch_size: int = 512,
    num_workers: int = 4,
):
    if not metacell_num and not metacell_frac:
        raise ValueError(
            "Must provide either a metacell_num or a metacell_frac to proceed"
        )

    print("=======Loading and Preprocessing Data=======")

    num_omics = len(data_path)
    print("Data of", num_omics, "omics in total")

    x_list = []
    sf_list = []
    raw_list = []
    adata_list = []
    num_list = []
    for i in range(num_omics):
        data_path = data_path[i]
        data_type = data_type[i]

        adata = sc.read_h5ad(data_path)
        x, sf, raw, adata = preprocess(adata, data_type)
        num = metacell_num if metacell_num else int(x.shape[0] * metacell_frac)

        x_list.append(x)
        sf_list.append(sf)
        raw_list.append(raw)
        adata_list.append(adata)
        num_list.append(num)

        print(data_path, "loaded with shape", list(x.shape))

    dataset = MetaQDataset(x_list, sf_list, raw_list)
    for n in num_list:
        if n < 512:
            batch_size = 128
        elif n > 10000 and batch_size <= 512:
            batch_size = 4096

    drop_last = (
        dataset.cell_num > batch_size * 4
    )  # Only drop last if we have plenty of cells

    if dataset.cell_num < batch_size:
        raise ValueError(
            f"Dataset has {dataset.cell_num} cells but batch size is {batch_size}. "
            f"Reduce batch size or use a larger dataset."
        )

    dataloader_train = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    dataloader_eval = DataLoader(
        dataset=dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    input_dims = [x.shape[1] for x in x_list]

    return adata_list, dataloader_train, dataloader_eval, input_dims


def compute_metacell(
    adata: ad.AnnData,
    meta_ids: np.ndarray,
):
    meta_ids = meta_ids.astype(int)
    nz_metacell = np.unique(meta_ids)

    data = adata.X
    data_meta = np.stack([data[meta_ids == i].mean(axis=0) for i in nz_metacell])
    metacell_adata = sc.AnnData(
        data_meta,
        obs=None,
        var=adata.var,  # copy variables from original anndata
    )

    return metacell_adata
