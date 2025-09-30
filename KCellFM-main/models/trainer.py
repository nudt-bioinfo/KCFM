import torch
from tqdm import tqdm
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, 
                            classification_report, confusion_matrix)
def load_pretrained_weights(model, pretrained_path, device,nlayers=6,fine_tune_layers=2):
    """加载预训练权重"""
    try:
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        
        loaded_layers = []
        not_loaded_layers = []
        shape_mismatch_layers = []
        
        pretrained_dict_filtered = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    pretrained_dict_filtered[k] = v
                    loaded_layers.append(k)
                else:
                    shape_mismatch_layers.append(f"{k} (expected {model_dict[k].shape}, got {v.shape})")
            else:
                not_loaded_layers.append(k)
        
        # 不加载分类头的权重
        # if 'cls_decoder.out_layer.weight' in pretrained_dict_filtered:
        #     del pretrained_dict_filtered['cls_decoder.out_layer.weight']
        # if 'cls_decoder.out_layer.bias' in pretrained_dict_filtered:
        #     del pretrained_dict_filtered['cls_decoder.out_layer.bias']
        
        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)
        nn.init.kaiming_normal_(model.cls_decoder.out_layer.weight, 
                              mode="fan_in", nonlinearity="relu")
        if model.cls_decoder.out_layer.bias is not None:
            nn.init.zeros_(model.cls_decoder.out_layer.bias)
        print("已重新初始化分类头参数。")
        
        # 3. 分层冻结/解冻设置
        # 冻结前 nlayers - fine_tune_layers 层
        for i in range(nlayers - fine_tune_layers):
            layer = model.mamba_encoder[i]
            for param in layer.parameters():
                param.requires_grad = False
            print(f"冻结层 {i}: 参数已冻结")
        
        # 解冻最后 fine_tune_layers 层
        for i in range(nlayers - fine_tune_layers, nlayers):
            layer = model.mamba_encoder[i]
            for param in layer.parameters():
                param.requires_grad = True
            print(f"解冻层 {i}: 参数可训练")
        print("\nSuccessfully loaded weights for the following layers:")
        for layer in loaded_layers:
            print(f"- {layer}")
            
        if shape_mismatch_layers:
            print("\nShape mismatch for these layers (not loaded):")
            for layer in shape_mismatch_layers:
                print(f"- {layer}")
                
        if not_loaded_layers:
            print("\nThese pretrained layers were not found in the model:")
            for layer in not_loaded_layers:
                print(f"- {layer}")
                
        print(f"\nTotal: {len(loaded_layers)} layers loaded successfully")
        
    except Exception as e:
        print(f"\nCould not load pretrained weights: {str(e)}")
    return model

def train_epoch(model, train_loader, optimizer, criterion, scaler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        src = batch['src'].to(device)
        values = batch['values'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        cell_types = batch['celltype'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            model_output = model(
                src=src,
                values=values,
                src_key_padding_mask=padding_mask
            )
            loss = criterion(model_output["cls_output"], cell_types)
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        batch_size_current = src.size(0)
        total_loss += loss.item() * batch_size_current
        total_samples += batch_size_current
        
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / total_samples if total_samples > 0 else 0

def test_epoch(model, test_loader,device):
    """评估模型在测试集上的性能"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            src = batch['src'].to(device)
            values = batch['values'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            cell_types = batch['celltype'].to(device)
            # 前向传播
            with torch.cuda.amp.autocast(enabled=True):
                model_output = model(
                    src=src,
                    values=values,
                    src_key_padding_mask=padding_mask
                )
            
            # 收集预测和标签
            preds = torch.argmax(model_output["cls_output"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(cell_types.cpu().numpy())
    return all_preds,all_labels

def get_h5ad_files(data_dir):
    """获取所有h5ad文件"""
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.endswith(".h5ad")]