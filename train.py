import numpy as np
from sklearn import metrics
import torch
import datasets
import models
from instrumentation import compute_metrics
import losses
import datetime
import os
from tqdm import tqdm

def run_train(P):

    dataset = datasets.get_data(P)
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True
        )


    model = models.ImageClassifier(P)
    
    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    linear_classifier_params = [param for param in list(model.linear_classifier.parameters()) if param.requires_grad]
    opt_params = [
        {'params': feature_extractor_params, 'lr' : P['lr']},
        {'params': linear_classifier_params, 'lr' : P['lr_mult'] * P['lr']}
    ]
  
    if P['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=P['lr'])
    elif P['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(opt_params, lr=P['lr'], momentum=0.9, weight_decay=0.001)
    
    # training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    bestmap_val = 0
    bestmap_test = 0

    for epoch in range(1, P['num_epochs']+1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
                y_true = np.zeros((len(dataset[phase]), P['num_classes']))
                batch_stack = 0

            
            with torch.set_grad_enabled(phase == 'train'):
                for batch in tqdm(dataloader[phase]):
                    # Move data to GPU
                    image = batch['image'].to(device, non_blocking=True)
                    label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                    label_vec_true = batch['label_vec_true'].clone().numpy()
                    idx = batch['idx']

                    # Forward pass
                    optimizer.zero_grad()

                    logits = model(image)
                   
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    preds = torch.sigmoid(logits)
                    
                    if phase == 'train':
                        loss_tensor, correction_idx = losses.compute_batch_loss(logits, label_vec_obs, P)
                        loss_tensor.backward()
                        optimizer.step()

                        if correction_idx is not None:
                            if P['perm_mod']: # permanent correction
                                dataset[phase].label_matrix_obs[idx[correction_idx[0].cpu()], correction_idx[1].cpu()] = 1.0

                    else:
                        preds_np = preds.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
                        y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
                        batch_stack += this_batch_size

            if phase != 'train':
                metrics = compute_metrics(y_pred, y_true)
                del y_pred
                del y_true
                map_val = metrics['map']
                
        P['llr_rel'] -= P['delta_rel']
                
        print(f"Epoch {epoch} : val mAP {map_val:.3f}")
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            
            print(f'Saving model weight for best val mAP {bestmap_val:.3f}')
            path = os.path.join(P['save_path'], 'bestmodel.pt')
            torch.save((model.state_dict(), P), path)
        
        elif bestmap_val - map_val > 3:
            print('Early stopped.')
            break
        
    # Test phase

    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)

    phase = 'test'
    model.eval()
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    batch_stack = 0
    with torch.set_grad_enabled(phase == 'train'):
        for batch in tqdm(dataloader[phase]):
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            idx = batch['idx']

            # Forward pass
            optimizer.zero_grad()

            logits = model(image)
                   
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
               
            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']

    print('Training procedure completed!')
    print(f'Test mAP : {map_test:.3f} when trained until epoch {bestmap_epoch}')
    