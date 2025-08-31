import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

from pretrain_model import MolecularPretrainModel, PretrainLoss, load_pretrained_weights
from pretrain_data import create_pretrain_dataloader
from model1 import BIN_Interaction_Flat


def train_pretrain_model(config):
    """预训练模型的训练函数"""
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = create_pretrain_dataloader(
        csv_path=config['data_path'],
        batch_size=config['batch_size'],
        max_len=config['max_drug_seq'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    print("创建预训练模型...")
    model = MolecularPretrainModel(**config).to(device)
    
    # 损失函数
    criterion = PretrainLoss(
        property_weight=config['property_weight'],
        mlm_weight=config['mlm_weight']
    )
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs']
    )
    
    # 训练循环
    model.train()
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        epoch_property_loss = 0.0
        epoch_mlm_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch in progress_bar:
            # 数据移到设备
            input_ids = batch['input_ids'].to(device)
            masked_input_ids = batch['masked_input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            properties = batch['properties'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            outputs = model(
                smiles_tokens=input_ids,
                attention_mask=attention_mask,
                masked_tokens=masked_input_ids
            )
            
            # 计算损失
            targets = {}
            if properties.numel() > 0:  # 如果有分子性质数据
                targets['properties'] = properties
            if config['use_mlm']:
                targets['masked_tokens'] = mlm_labels
            
            loss_dict = criterion(outputs, targets)
            total_loss = loss_dict['total_loss']
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            
            # 统计损失
            epoch_loss += total_loss.item()
            if 'property_loss' in loss_dict:
                epoch_property_loss += loss_dict['property_loss'].item()
            if 'mlm_loss' in loss_dict:
                epoch_mlm_loss += loss_dict['mlm_loss'].item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # 调整学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        avg_property_loss = epoch_property_loss / num_batches if epoch_property_loss > 0 else 0
        avg_mlm_loss = epoch_mlm_loss / num_batches if epoch_mlm_loss > 0 else 0
        
        print(f"Epoch {epoch+1}:")
        print(f"  平均总损失: {avg_loss:.4f}")
        if avg_property_loss > 0:
            print(f"  平均性质损失: {avg_property_loss:.4f}")
        if avg_mlm_loss > 0:
            print(f"  平均MLM损失: {avg_mlm_loss:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['save_path'])
            print(f"  保存最佳模型到: {config['save_path']}")
        
        # 定期保存检查点
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = config['save_path'].replace('.pth', f'_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  保存检查点到: {checkpoint_path}")
    
    print("预训练完成！")
    return model


def test_pretrained_model(config):
    """测试预训练模型是否能正确加载到主模型"""
    
    print("测试预训练权重加载...")
    
    # 创建主模型（BIN_Interaction_Flat）
    main_model = BIN_Interaction_Flat(
        hidden_size=config['emb_size'],
        input_dim_drug=config['input_dim_drug'],
        max_drug_seq=config['max_drug_seq'],
        emb_size=config['emb_size'],
        dropout_rate=config['dropout_rate'],
        intermediate_size=config['intermediate_size'],
        num_attention_heads=config['num_attention_heads'],
        attention_probs_dropout_prob=config['attention_probs_dropout_prob'],
        hidden_dropout_prob=config['hidden_dropout_prob'],
        # 其他必要参数...
    )
    
    # 加载预训练权重
    if os.path.exists(config['save_path']):
        load_pretrained_weights(main_model, config['save_path'])
        print("预训练权重加载测试成功！")
    else:
        print(f"预训练模型文件不存在: {config['save_path']}")


def main():
    parser = argparse.ArgumentParser(description='分子预训练')
    parser.add_argument('--data_path', type=str, default='pretrain_data.csv', help='预训练数据文件路径')
    parser.add_argument('--save_path', type=str, default='pretrained_molecular_model.pth', help='模型保存路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--test_only', action='store_true', help='仅测试权重加载')
    
    args = parser.parse_args()
    
    # 获取词汇表大小
    try:
        sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
        vocab_size = len(sub_csv)
        print(f"词汇表大小: {vocab_size}")
    except FileNotFoundError:
        print("未找到词汇表文件，使用默认大小")
        vocab_size = 8889  # 默认值
    
    # 配置参数
    config = {
        # 数据配置
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'num_workers': 4,
        
        # 模型配置（与原模型保持一致）
        'input_dim_drug': vocab_size,
        'max_drug_seq': 50,
        'emb_size': 256,
        'intermediate_size': 512,
        'num_attention_heads': 8,
        'attention_probs_dropout_prob': 0.1,
        'hidden_dropout_prob': 0.1,
        'dropout_rate': 0.1,
        'n_layer': 2,
        'num_properties': 17,  # prop_0 到 prop_16，共17个性质
        'use_mlm': True,
        
        # 训练配置
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'max_grad_norm': 1.0,
        'property_weight': 1.0,
        'mlm_weight': 0.5,
        
        # 保存配置
        'save_path': args.save_path,
        'save_interval': 10,
    }
    
    if args.test_only:
        test_pretrained_model(config)
    else:
        # 检查数据文件
        if not os.path.exists(args.data_path):
            print(f"数据文件不存在: {args.data_path}")
            return
        
        # 开始训练
        model = train_pretrain_model(config)
        
        # 测试权重加载
        test_pretrained_model(config)


if __name__ == "__main__":
    main()
