#!/bin/bash

# LFS下载监控脚本
LOG_FILE="/root/lanyun-tmp/lfs_monitor.log"
DATA_DIR="/root/lanyun-tmp/data_download/mrag/data"

echo "开始监控LFS下载进度..." | tee -a $LOG_FILE
echo "监控时间: $(date)" | tee -a $LOG_FILE
echo "数据目录: $DATA_DIR" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# 启动LFS下载
cd /root/lanyun-tmp/data_download/mrag
echo "启动LFS下载..." | tee -a $LOG_FILE
nohup git lfs pull > /root/lanyun-tmp/lfs_download.log 2>&1 &
LFS_PID=$!
echo "LFS进程ID: $LFS_PID" | tee -a $LOG_FILE

# 监控循环
while true; do
    echo "----------------------------------------" | tee -a $LOG_FILE
    echo "检查时间: $(date)" | tee -a $LOG_FILE
    
    # 检查进程状态
    if ps -p $LFS_PID > /dev/null 2>&1; then
        echo "LFS进程运行中 (PID: $LFS_PID)" | tee -a $LOG_FILE
    else
        echo "LFS进程已结束" | tee -a $LOG_FILE
        break
    fi
    
    # 检查文件大小
    echo "文件大小检查:" | tee -a $LOG_FILE
    ls -lh $DATA_DIR/*.parquet | tee -a $LOG_FILE
    
    # 计算总大小
    TOTAL_SIZE=$(du -sh $DATA_DIR | cut -f1)
    echo "目录总大小: $TOTAL_SIZE" | tee -a $LOG_FILE
    
    # 检查是否有文件超过134字节
    LARGE_FILES=$(find $DATA_DIR -name "*.parquet" -size +134c | wc -l)
    echo "已下载文件数: $LARGE_FILES/7" | tee -a $LOG_FILE
    
    if [ $LARGE_FILES -eq 7 ]; then
        echo "所有文件下载完成！" | tee -a $LOG_FILE
        break
    fi
    
    # 等待30秒
    echo "等待30秒..." | tee -a $LOG_FILE
    sleep 30
done

echo "监控结束: $(date)" | tee -a $LOG_FILE
