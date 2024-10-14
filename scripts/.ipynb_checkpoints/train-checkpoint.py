import argparse
import torch
import torch.optim as optim
from transformers import AutoTokenizer
from custom_model import UnifiedLlamaModel, BiasRemoval, ContrastiveLoss  # 确保 custom_model.py 包含您的模型定义
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os

def setup_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置随机种子以确保结果可复现
    torch.manual_seed(args.seed)

    # 加载模型
    model = UnifiedLlamaModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto" if args.quantization else None,
    ).to(device)

    if args.use_peft:
        from peft import get_peft_model, PeftModel
        if args.peft_checkpoint:
            model = PeftModel.from_pretrained(model, args.peft_checkpoint, is_trainable=True)
        else:
            print("PEFT configuration is missing or incorrect. Exiting...")
            return

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = ContrastiveLoss(margin=args.margin).to(device)

    return model, optimizer, criterion, device

def train_model(model, optimizer, criterion, device, dataloader, args):
    scaler = GradScaler()
    model.train()
    
    for epoch in tqdm(range(args.epochs)):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # 使用 autocast 处理
            with torch.cuda.amp.autocast():
                input_text_ids = batch['input_text_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                original_input_ids = batch['original_input_ids'].to(device)
                original_attention_mask = batch['original_attention_mask'].to(device)
                bias_input_ids = batch['bias_input_ids'].to(device)
                bias_attention_mask = batch['bias_attention_mask'].to(device)

                # 获取 embeddings
                bias_embeddings = model(bias_input_ids, bias_attention_mask, task="rank")
                original_embeddings = model(original_input_ids, original_attention_mask, task="rank")

                # 生成任务的 logits
                predicted_labels = model(input_text_ids, text_attention_mask, task="label_generation")

                # 计算对比损失
                loss = criterion(original_embeddings, original_embeddings, bias_embeddings)

            # 反向传播和优化
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    # 保存模型
    if args.save_model:
        model.save_pretrained(args.output_dir)
        print(f"Model saved at {args.output_dir}")

def main():
    # Argument parser to take inputs from command line
    parser = argparse.ArgumentParser(description="Train UnifiedLlamaModel with Contrastive Learning")

    # Add model and training related arguments
    parser.add_argument("--model_name", type=str, default="facebook/llama", help="Pretrained model name or path")
    parser.add_argument("--task", type=str, default="qa", help="Task to perform: qa or contrastive")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for contrastive loss")
    parser.add_argument("--use_peft", action="store_true", help="Whether to use PEFT")
    parser.add_argument("--peft_checkpoint", type=str, help="Path to PEFT checkpoint if using PEFT")
    parser.add_argument("--quantization", action="store_true", help="Enable quantization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_model", action="store_true", help="Whether to save the trained model")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Directory to save the model")
    
    args = parser.parse_args()

    # Setup training
    model, optimizer, criterion, device = setup_training(args)

    # Mock dataloader (Replace with actual dataloader)
    dataloader = []  # Replace with your dataset loader

    # Start training
    train_model(model, optimizer, criterion, device, dataloader, args)

if __name__ == "__main__":
    main()
