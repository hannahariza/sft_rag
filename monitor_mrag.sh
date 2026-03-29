#!/bin/bash

# mRAG_eval 监控脚本
echo "=== mRAG_eval 后台运行监控 ==="
echo "按 Ctrl+C 退出监控"
echo ""

# 获取最新的日志文件
LATEST_LOG=$(ls -t /root/lanyun-tmp/logs/mRAG_eval_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到mRAG_eval日志文件"
    exit 1
fi

echo "📋 监控日志文件: $LATEST_LOG"
echo ""

# 检查进程是否在运行
check_process() {
    if pgrep -f "qwen25vl_mRAG_eval.py" > /dev/null; then
        echo "✅ mRAG_eval 进程正在运行"
        return 0
    else
        echo "❌ mRAG_eval 进程未运行"
        return 1
    fi
}

# 显示GPU使用情况
show_gpu() {
    echo "🖥️  GPU使用情况:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        echo "   GPU $line"
    done
}

# 显示最新日志
show_latest_log() {
    echo "📝 最新日志 (最后10行):"
    tail -10 "$LATEST_LOG" 2>/dev/null | sed 's/^/   /'
}

# 显示进度统计
show_progress() {
    if [ -f "$LATEST_LOG" ]; then
        # 统计处理的问题数量
        PROCESSED=$(grep -c "成功处理问题" "$LATEST_LOG" 2>/dev/null || echo "0")
        TOTAL="50"  # 根据您的数据集调整
        
        echo "📊 进度统计:"
        echo "   已处理问题: $PROCESSED/$TOTAL"
        
        if [ "$PROCESSED" -gt 0 ]; then
            PERCENTAGE=$((PROCESSED * 100 / TOTAL))
            echo "   完成进度: $PERCENTAGE%"
        fi
    fi
}

# 主监控循环
while true; do
    clear
    echo "=== mRAG_eval 监控 - $(date) ==="
    echo ""
    
    check_process
    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠️  进程已停止，监控结束"
        break
    fi
    
    echo ""
    show_gpu
    echo ""
    show_progress
    echo ""
    show_latest_log
    echo ""
    echo "🔄 5秒后刷新... (按 Ctrl+C 退出)"
    
    sleep 5
done






