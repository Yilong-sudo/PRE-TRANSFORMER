import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import codecs
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from subword_nmt.apply_bpe import BPE
import random


# 使用与原dataset相同的BPE处理
vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))


def drug2emb_encoder(x, max_d=50):
    """与原dataset相同的SMILES编码函数"""
    t1 = dbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d
    return i, np.asarray(input_mask)


class ZincPretrainDataset(Dataset):
    """ZINC数据集预训练数据集"""
    
    def __init__(self, csv_path, max_len=50, mlm_probability=0.15):
        """
        Args:
            csv_path: ZINC CSV文件路径
            max_len: 最大序列长度（与原dataset一致）
            mlm_probability: 掩码语言模型的掩码概率
        """
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        
        # 根据实际pretrain_data文件的列名调整
        available_columns = list(self.data.columns)
        print(f"可用的列: {available_columns}")
        
        # 提取分子性质列（prop_0 到 prop_16，共17个性质）
        self.property_columns = []
        for col in available_columns:
            if col.startswith('prop_'):
                self.property_columns.append(col)
        
        print(f"检测到 {len(self.property_columns)} 个分子性质特征: {self.property_columns[:5]}{'...' if len(self.property_columns) > 5 else ''}")
        
        # 只保留有完整性质数据的样本
        self.data = self.data.dropna(subset=['smiles'] + self.property_columns)
        
        # 标准化分子性质
        if self.property_columns:
            self.scaler = StandardScaler()
            self.properties = self.scaler.fit_transform(self.data[self.property_columns].values)
        else:
            self.properties = None
        
        print(f"加载了 {len(self.data)} 个分子样本")
        if self.properties is not None:
            print(f"分子性质维度: {self.properties.shape[1]}")
            print(f"使用的性质列: {self.property_columns}")
        else:
            print("未找到可用的分子性质列，仅使用MLM任务")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['smiles']
        
        # 使用与原dataset相同的SMILES编码方式
        tokens, attention_mask = drug2emb_encoder(smiles, self.max_len)
        
        # 获取分子性质
        if self.properties is not None:
            properties = self.properties[idx]
        else:
            properties = np.array([])  # 空数组
        
        # 掩码语言模型处理
        masked_tokens, mlm_labels = self.create_mlm_data(tokens)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'masked_input_ids': torch.tensor(masked_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'properties': torch.tensor(properties, dtype=torch.float),
            'smiles': smiles
        }
    
    def create_mlm_data(self, tokens):
        """创建掩码语言模型数据"""
        masked_tokens = tokens.copy()
        mlm_labels = np.full(len(tokens), -100)  # -100表示不计算损失
        
        vocab_size = len(words2idx_d)
        
        for i, token in enumerate(tokens):
            if token == 0:  # 跳过填充token
                continue
                
            if random.random() < self.mlm_probability:
                mlm_labels[i] = token  # 保存原始token作为标签
                
                prob = random.random()
                if prob < 0.8:
                    # 80%替换为一个特殊的mask token（使用词汇表中的一个特殊索引）
                    masked_tokens[i] = vocab_size - 1 if vocab_size > 1 else 1
                elif prob < 0.9:
                    # 10%随机替换
                    masked_tokens[i] = random.randint(1, vocab_size - 1)
                # 10%保持不变
        
        return masked_tokens, mlm_labels


def create_pretrain_dataloader(csv_path, batch_size=32, max_len=50, num_workers=4):
    """创建预训练数据加载器"""
    dataset = ZincPretrainDataset(csv_path, max_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def prepare_zinc_data(zinc_csv_path, output_path=None):
    """预处理ZINC数据"""
    df = pd.read_csv(zinc_csv_path)
    
    # 检查必要的列
    required_columns = ['smiles']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告: 缺少列 {missing_columns}")
        print(f"可用的列: {list(df.columns)}")
        return None
    
    # 移除无效的SMILES
    df = df.dropna(subset=['smiles'])
    df = df[df['smiles'].str.len() > 0]
    
    print(f"预处理后保留 {len(df)} 个分子")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"处理后的数据保存到: {output_path}")
    
    return df


if __name__ == "__main__":
    # 测试数据加载
    csv_path = "pretrain_data.csv"  # 您的预训练数据文件
    
    try:
        # 预处理数据
        df = prepare_zinc_data(csv_path)
        
        if df is not None:
            # 创建数据加载器
            dataloader = create_pretrain_dataloader(
                csv_path, 
                batch_size=8, 
                max_len=50  # 与原dataset一致
            )
            
            # 测试数据加载
            for batch in dataloader:
                print("批次信息:")
                print(f"  input_ids: {batch['input_ids'].shape}")
                print(f"  attention_mask: {batch['attention_mask'].shape}")
                print(f"  properties: {batch['properties'].shape}")
                print(f"  示例SMILES: {batch['smiles'][0]}")
                
                # 测试编码
                tokens, mask = drug2emb_encoder(batch['smiles'][0])
                print(f"  编码tokens: {tokens[:10]}...")
                print(f"  注意力掩码: {mask[:10]}...")
                break
                
    except FileNotFoundError:
        print("请确保zinc_250k.csv文件存在于当前目录")
    except Exception as e:
        print(f"数据加载测试出错: {e}")
        import traceback
        traceback.print_exc()
