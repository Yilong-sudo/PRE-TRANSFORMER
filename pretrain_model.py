import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from model1 import Embeddings, Encoder_MultipleLayers


class MolecularPretrainModel(nn.Module):
    """分子预训练模型 - 用于学习分子表示"""
    
    def __init__(self, **config):
        super(MolecularPretrainModel, self).__init__()
        
        # 基础配置（与原模型保持一致）
        self.input_dim_drug = config['input_dim_drug']
        self.max_d = config.get('max_drug_seq', 50)  # 与原dataset一致
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']
        self.n_layer = config.get('n_layer', 2)
        
        # Encoder配置
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        
        # 分子性质预测任务数量（根据pretrain_data调整）
        self.num_properties = config.get('num_properties', 17)  # prop_0 到 prop_16，共17个性质
        
        # 核心编码器模块（与主模型完全相同）
        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.d_encoder = Encoder_MultipleLayers(
            self.n_layer, self.hidden_size, self.intermediate_size,
            self.num_attention_heads, self.attention_probs_dropout_prob,
            self.hidden_dropout_prob
        )
        
        # 预训练特定的预测头
        self.property_predictor = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.emb_size // 2, self.num_properties)
        )
        
        # 可选的掩码语言模型任务
        self.use_mlm = config.get('use_mlm', True)
        if self.use_mlm:
            self.mlm_head = nn.Linear(self.emb_size, self.input_dim_drug)
    
    def forward(self, smiles_tokens, attention_mask, masked_tokens=None):
        """
        Args:
            smiles_tokens: [batch_size, seq_len] SMILES序列的token
            attention_mask: [batch_size, seq_len] 注意力掩码
            masked_tokens: [batch_size, seq_len] MLM任务的掩码tokens（可选）
        """
        # 扩展注意力掩码（与原模型相同的处理方式）
        ex_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        ex_mask = (1.0 - ex_mask) * -10000.0
        
        # 分子嵌入
        mol_emb = self.demb(smiles_tokens)
        
        # 编码器处理
        encoded_layers = self.d_encoder(mol_emb.float(), ex_mask.float())
        
        # 分子级别的表示（与原模型中d_aug1的计算方式相同）
        mol_repr = torch.sum(encoded_layers, dim=1)  # 求和池化
        
        # 分子性质预测
        property_pred = self.property_predictor(mol_repr)
        
        outputs = {'property_pred': property_pred, 'mol_repr': mol_repr}
        
        # 掩码语言模型预测（可选）
        if self.use_mlm and masked_tokens is not None:
            # 使用masked tokens进行MLM预测
            masked_emb = self.demb(masked_tokens)
            masked_encoded = self.d_encoder(masked_emb.float(), ex_mask.float())
            mlm_pred = self.mlm_head(masked_encoded)
            outputs['mlm_pred'] = mlm_pred
            
        return outputs
    
    def get_molecular_representation(self, smiles_tokens, attention_mask):
        """获取分子表示（用于下游任务）"""
        with torch.no_grad():
            outputs = self.forward(smiles_tokens, attention_mask)
            return outputs['mol_repr']


class PretrainLoss(nn.Module):
    """预训练损失函数"""
    
    def __init__(self, property_weight=1.0, mlm_weight=0.5):
        super(PretrainLoss, self).__init__()
        self.property_weight = property_weight
        self.mlm_weight = mlm_weight
        
        # 不同性质可能需要不同的损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: 模型输出
            targets: 目标值字典
                - 'properties': [batch_size, num_properties] 分子性质
                - 'masked_tokens': [batch_size, seq_len] MLM目标（可选）
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 分子性质预测损失
        if 'properties' in targets:
            prop_loss = self.mse_loss(outputs['property_pred'], targets['properties'])
            total_loss += self.property_weight * prop_loss
            loss_dict['property_loss'] = prop_loss
        
        # 掩码语言模型损失
        if 'mlm_pred' in outputs and 'masked_tokens' in targets:
            mlm_pred = outputs['mlm_pred'].view(-1, outputs['mlm_pred'].size(-1))
            mlm_targets = targets['masked_tokens'].view(-1)
            mlm_loss = self.ce_loss(mlm_pred, mlm_targets)
            total_loss += self.mlm_weight * mlm_loss
            loss_dict['mlm_loss'] = mlm_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict


def load_pretrained_weights(main_model, pretrain_model_path):
    """将预训练权重加载到主模型中"""
    pretrain_state = torch.load(pretrain_model_path, map_location='cpu')
    
    # 获取主模型的state_dict
    main_state = main_model.state_dict()
    
    # 映射预训练权重到主模型
    weight_mapping = {}
    
    # 嵌入层权重映射
    weight_mapping.update({
        'demb.word_embeddings.weight': 'demb.word_embeddings.weight',
        'demb.position_embeddings.weight': 'demb.position_embeddings.weight',
        'demb.LayerNorm.gamma': 'demb.LayerNorm.gamma',
        'demb.LayerNorm.beta': 'demb.LayerNorm.beta'
    })
    
    # 编码器层权重映射
    for i in range(len(main_model.d_encoder.layer)):
        layer_prefix = f'd_encoder.layer.{i}'
        param_names = [
            'attention.self.query.weight', 'attention.self.query.bias',
            'attention.self.key.weight', 'attention.self.key.bias',
            'attention.self.value.weight', 'attention.self.value.bias',
            'attention.output.dense.weight', 'attention.output.dense.bias',
            'attention.output.LayerNorm.gamma', 'attention.output.LayerNorm.beta',
            'intermediate.dense.weight', 'intermediate.dense.bias',
            'output.dense.weight', 'output.dense.bias',
            'output.LayerNorm.gamma', 'output.LayerNorm.beta'
        ]
        
        for param_name in param_names:
            full_name = f'{layer_prefix}.{param_name}'
            weight_mapping[full_name] = full_name
    
    # 加载权重
    loaded_count = 0
    for pretrain_key, main_key in weight_mapping.items():
        if pretrain_key in pretrain_state and main_key in main_state:
            if main_state[main_key].shape == pretrain_state[pretrain_key].shape:
                main_state[main_key] = pretrain_state[pretrain_key]
                loaded_count += 1
                print(f"✓ 加载: {pretrain_key}")
            else:
                print(f"✗ 形状不匹配: {pretrain_key} - 预训练:{pretrain_state[pretrain_key].shape} vs 主模型:{main_state[main_key].shape}")
        else:
            if pretrain_key not in pretrain_state:
                print(f"✗ 预训练模型中未找到: {pretrain_key}")
            if main_key not in main_state:
                print(f"✗ 主模型中未找到: {main_key}")
    
    main_model.load_state_dict(main_state)
    print(f"\n预训练权重加载完成！成功加载 {loaded_count}/{len(weight_mapping)} 个参数")
    return main_model


if __name__ == "__main__":
    # 根据原模型配置设置参数
    sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
    vocab_size = len(sub_csv)
    
    config = {
        'input_dim_drug': vocab_size,  # 使用实际的词汇表大小
        'max_drug_seq': 50,            # 与原dataset一致
        'emb_size': 256,
        'intermediate_size': 512,
        'num_attention_heads': 8,
        'attention_probs_dropout_prob': 0.1,
        'hidden_dropout_prob': 0.1,
        'dropout_rate': 0.1,
        'n_layer': 2,
        'num_properties': 17,           # prop_0 到 prop_16，共17个性质
        'use_mlm': True
    }
    
    print(f"词汇表大小: {vocab_size}")
    print(f"配置参数: {config}")
    
    # 创建模型
    model = MolecularPretrainModel(**config)
    
    # 测试前向传播
    batch_size, seq_len = 4, 50
    smiles_tokens = torch.randint(0, config['input_dim_drug'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    masked_tokens = torch.randint(0, config['input_dim_drug'], (batch_size, seq_len))
    
    outputs = model(smiles_tokens, attention_mask, masked_tokens)
    print(f"\n模型测试:")
    print(f"分子表示形状: {outputs['mol_repr'].shape}")
    print(f"性质预测形状: {outputs['property_pred'].shape}")
    if 'mlm_pred' in outputs:
        print(f"MLM预测形状: {outputs['mlm_pred'].shape}")
