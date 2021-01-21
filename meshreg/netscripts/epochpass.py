import os
import pickle

from tqdm import tqdm
import torch

from meshreg.visualize import samplevis, evalvis
from meshreg.visualize import consistdisplay
from meshreg.netscripts.monitor import MetricMonitor
from meshreg.netscripts.metrics import evaluate


def epoch_pass(
    loader,
    model,
    train=False,
    optimizer=None,
    scheduler=None,
    epoch=0,
    img_folder=None,
    fig=None,
    display_freq=10,
    epoch_display_freq=1,
    lr_decay_gamma=0,
    freeze_batchnorm=True,
    monitor=None,
    save_inference=False,
):
    if train:
        prefix = "train"
        if not freeze_batchnorm:
            model.train()
    else:
        prefix = "val"
        model.eval()

    render_step = 0
    inference_results = []

    # Loop over dataset
    for batch_idx, batch in enumerate(tqdm(loader, desc="batch")):
        if 'compute_pnp' in dir(model):
            model.compute_pnp = not train or (batch_idx % display_freq == 0 and epoch % epoch_display_freq == 0)

        # Compute outputs and losses
        if train:
            loss, results, losses = model(batch)
            # Optimize model
            if torch.isnan(loss):
                raise ValueError(f"Loss made of {losses} became nan!")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss, results, losses = model(batch)
                if save_inference:
                    inference_results.append(results)

        # Update metrics
        if monitor is not None:
            # Create loss dict, add _loss suffix where necessary
            loss_dict = {}
            for loss_name, loss_val in losses.items():
                if not (loss_name.startswith("loss_") or loss_name.endswith("_loss")):
                    loss_name = "loss_{}".format(loss_name)
                if loss_val is not None:
                    if isinstance(loss_val, torch.Tensor):
                        loss_val = loss_val.cpu().detach().numpy()
                    loss_dict[loss_name] = loss_val
            monitor.add(prefix, epoch + 1, loss_dict)
            monitor.add(prefix, epoch + 1, evaluate(batch, results))

        # Visualize sample outputs
        if batch_idx % display_freq == 0 and epoch % epoch_display_freq == 0:
            img_filepath = f"{prefix}_epoch{epoch:04d}_batch{batch_idx:06d}.png"
            save_img_path = os.path.join(img_folder, img_filepath)
            samplevis.sample_vis(batch, results, fig=fig, save_img_path=save_img_path)

    if save_inference:
        with open("inference.pkl", "wb") as f:
            pickle.dump(inference_results, f)

    if lr_decay_gamma and scheduler is not None:
        scheduler.step()

    save_dict = {}
    return save_dict
