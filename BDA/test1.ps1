# 定义参数数组
$epochs = @(10)
$K_values = @(5)

# 使用嵌套循环来迭代所有参数组合
foreach ($K in $K_values) {
    foreach ($epoch in $epochs) {
        # 构建并执行命令
        $command = "python BDA.py --flag=`"Hybrid`" --logdir=`"logs1_Hybrid`""
        Write-Host "Executing command: $command"
        Invoke-Expression $command
    }
}