import json
import glob
import os

def fix_notebooks():
    # 查找当前目录下所有以 torch- 开头的 ipynb 文件
    files = glob.glob("torch-*.ipynb")
    if not files:
        print("未找到 torch- 开头的 ipynb 文件！请确保脚本放在文件同一目录下。")
        return
        
    for file_name in files:
        with open(file_name, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        modified = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
                
            # 将 cell 的内容合并成一个完整字符串，方便替换
            source = "".join(cell['source'])
            original_source = source
            
            # 1. 修复 TensorDataset CPU->GPU 搬运问题 (适用于所有算法)
            source = source.replace("torch.from_numpy(self.X_train)", "self.X_train_t")
            source = source.replace("torch.from_numpy(self.y_train)", "self.y_train_t")
            
            # 2. 修复梯度计算时强行拉回 CPU 的致命瓶颈 (FedAvg, FedProx, RetroFL)
            source = source.replace(
                "grads = [p.grad.detach().cpu().clone() for p in self.model.parameters()]",
                "grads = [p.grad.detach().clone() for p in self.model.parameters()]"
            )
            
            # 3. 修复 Scaffold 中的克隆灾难 1：内层循环的梯度和 c_i 大量冗余克隆
            scaffold_bad_loop = """            corrected_grads = []
            for p, ci, cg in zip(self.model.parameters(), self.c_i, c_global):
                target_device = p.device
                grad = p.grad.detach().clone().to(target_device)
                ci = ci.detach().clone().to(target_device)
                cg = cg.detach().clone().to(target_device)
                corrected_grads.append(grad - ci + cg)

            with torch.no_grad():
                for p, g_corr in zip(self.model.parameters(), corrected_grads):
                    p.sub_(current_lr * g_corr)"""
                    
            scaffold_good_loop = """            with torch.no_grad():
                for p, ci, cg in zip(self.model.parameters(), self.c_i, c_global):
                    # 直接进行原地计算，不产生任何临时 Tensor 拷贝
                    corrected_grad = p.grad - ci + cg
                    p.sub_(current_lr * corrected_grad)"""
                    
            if scaffold_bad_loop in source:
                source = source.replace(scaffold_bad_loop, scaffold_good_loop)
                
            # 4. 修复 Scaffold 中的克隆灾难 2：控制变量增量的计算
            scaffold_bad_c = """        new_c_i = []
        delta_c_i = []
        for ci_old, c_g, w_b, w_a in zip(self.c_i, c_global, w_before, w_after):
            ci_old = ci_old.detach().clone().to(self.device)
            c_g = c_g.detach().clone().to(self.device)
            w_b = w_b.detach().clone().to(self.device)
            w_a = w_a.detach().clone().to(self.device)

            ci_new = ci_old - c_g + scale * (w_b - w_a)
            ci_new = ci_new.detach().clone().to(self.device)

            new_c_i.append(ci_new)
            delta_c_i.append((ci_new - ci_old).detach().clone().to(self.device))"""
            
            scaffold_good_c = """        new_c_i = []
        delta_c_i = []
        for ci_old, c_g, w_b, w_a in zip(self.c_i, c_global, w_before, w_after):
            ci_new = ci_old - c_g + scale * (w_b - w_a)
            new_c_i.append(ci_new.detach())
            delta_c_i.append((ci_new - ci_old).detach())"""
            
            if scaffold_bad_c in source:
                source = source.replace(scaffold_bad_c, scaffold_good_c)
                
            if source != original_source:
                modified = True
                # 将字符串切分回 Jupyter 要求的行列表格式
                lines = [line + '\n' for line in source.split('\n')]
                lines[-1] = lines[-1].rstrip('\n') 
                if not lines[-1]:
                    lines.pop()
                cell['source'] = lines
                
        if modified:
            new_name = file_name.replace('.ipynb', '_fixed.ipynb')
            with open(new_name, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"已修复并生成高效版本: {new_name}")

if __name__ == '__main__':
    print("开始修复无意义张量搬运...")
    fix_notebooks()
    print("修复完成！你可以使用 _fixed.ipynb 后缀的文件进行训练了。")