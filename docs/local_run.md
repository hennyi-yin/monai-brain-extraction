# 本机执行

项目目录：`C:\Users\25696\Downloads\MONAI`。

已在 Ubuntu-24.04 WSL 中创建独立环境，不改动系统 Python：

- 训练环境：`/home/hengyi/.venvs/monai-brain`
- FSL BET 环境：`/home/hengyi/.venvs/fsl-bet`

在 PowerShell 打开 WSL 后运行：

```bash
wsl -d Ubuntu-24.04
cd /mnt/c/Users/25696/Downloads/MONAI
source /home/hengyi/.venvs/monai-brain/bin/activate
export FSLDIR=/home/hengyi/.venvs/fsl-bet
export PATH="$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ
bash scripts/run_project.sh
```

训练日志：`logs/training.csv`。断线/中断后使用同一命令，已有检查点时自动继续。不要修改冻结配置或划分。最终 `python evaluate.py` 生成逐例表、统计、三张结果图并自动更新 README。

数据下载来源为 NFBS 官方项目链接。原始图、缓存、权重及分割体积均由 `.gitignore` 排除。

FSL `-R` 使用系统 `dc` 计算器，本机已安装。若在新 Ubuntu 环境复现，需要 `sudo apt-get install dc`。`bash scripts/bootstrap_wsl.sh` 可创建独立训练/FSL 环境；其中 FSL 使用已记录的精确包构建。
