import torch
from torch.utils.data import DataLoader
from alive_progress import alive_bar


from .model import MetaQ
from .engine import inference, train_one_epoch, warm_one_epoch


def train_model(
    net: MetaQ,
    data_type: str,
    dataloader_train: DataLoader,
    dataloader_eval: DataLoader,
    omics_num: int,
    train_epoch: int = 300,
    codebook_init: str = "Random",
    converge_threshold: int = 10,
    device: str = "cuda",
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    epsilon: float = 1e-5,
):
    """Trains the metaq model on the dataset"""

    # Move the network to the required device
    net.to(device)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    print("======= Training Start =======")

    with alive_bar(train_epoch, enrich_print=False) as bar:
        loss_rec_his = loss_vq_his = 1e7
        stable_epochs = 0
        if codebook_init == "Random":
            warm_epochs = 0
        else:
            # For Kmeans and Geometric initialization
            warm_epochs = min(50, int(train_epoch * 0.2))
        for epoch in range(train_epoch):
            bar()
            if epoch < warm_epochs:
                warm_one_epoch(
                    model=net,
                    data_types=data_type,
                    dataloader=dataloader_train,
                    optimizer=optimizer,
                    epoch=epoch,
                    device=device,
                )
            elif epoch == warm_epochs:
                embeds, ids, _, _, _ = inference(
                    model=net,
                    data_types=data_type,
                    data_loader=dataloader_eval,
                    device=device,
                )
                net.quantizer.init_codebook(embeds, method=codebook_init)
                if omics_num == 1:
                    net.copy_decoder_q()
            else:
                loss_rec, loss_vq = train_one_epoch(
                    model=net,
                    data_types=data_type,
                    dataloader=dataloader_train,
                    optimizer=optimizer,
                    epoch=epoch,
                    device=device,
                )
                converge = (abs(loss_vq_his - loss_vq) <= epsilon) and (
                    abs(loss_rec_his - loss_rec) <= epsilon
                )
                if converge:
                    stable_epochs += 1
                    if stable_epochs >= converge_threshold:
                        print("Early Stopping.")
                        break
                else:
                    stable_epochs = 0
                    loss_rec_his = loss_rec
                    loss_vq_his = loss_vq

    print("======= Training Done =======")
