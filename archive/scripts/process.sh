#!/bin/bash

# GalaxyAI Content Analysis Script
# 用于对捕获的内容进行深度分析并生成报告

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
ANALYSIS_DIR="$PROJECT_ROOT/analysis"

# 加载配置
if [[ -f "$CONFIG_DIR/config.env" ]]; then
    source "$CONFIG_DIR/config.env"
fi

# 检查必需的环境变量
check_config() {
    if [[ -z "$OPENROUTER_API_KEY" ]]; then
        echo "❌ 错误: 未设置 OPENROUTER_API_KEY 环境变量"
        echo "请在 $CONFIG_DIR/config.env 中设置 API 密钥"
        exit 1
    fi
}

# 生成分析文件名
generate_filename() {
    local content="$1"
    local slug=$(echo "$content" | head -c 30 | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g')
    local date=$(date "+%Y-%m-%d")
    echo "${slug}-analysis-${date}.md"
}

# 调用OpenRouter API进行分析
call_openrouter_api() {
    local content="$1"
    local analysis_prompt="$2"
    
    local json_payload=$(jq -n \
        --arg model "${OPENROUTER_MODEL:-anthropic/claude-3.5-sonnet}" \
        --arg prompt "$analysis_prompt" \
        --arg content "$content" \
        '{
            model: $model,
            messages: [
                {
                    role: "user",
                    content: ($prompt + "\n\n需要分析的内容：\n" + $content)
                }
            ],
            max_tokens: 4000,
            temperature: 0.7
        }')
    
    echo "🤖 正在调用 OpenRouter API 进行深度分析..."
    echo "模型: ${OPENROUTER_MODEL:-anthropic/claude-3.5-sonnet}"
    
    local response=$(curl -s -X POST "https://openrouter.ai/api/v1/chat/completions" \
        -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        -H "Content-Type: application/json" \
        -H "HTTP-Referer: https://github.com/galaxyai" \
        -H "X-Title: GalaxyAI Analysis" \
        -d "$json_payload")
    
    # 检查API调用是否成功
    if [[ $? -ne 0 ]]; then
        echo "❌ API调用失败"
        exit 1
    fi
    
    # 提取分析结果
    local analysis=$(echo "$response" | jq -r '.choices[0].message.content // empty')
    
    if [[ -z "$analysis" ]]; then
        echo "❌ API响应无效:"
        echo "$response" | jq '.'
        exit 1
    fi
    
    echo "$analysis"
}

# 构建分析提示词
build_analysis_prompt() {
    cat << 'EOF'
你是一个专业的技术分析师和战略顾问。请对用户提供的内容进行深度分析，生成一份结构化的分析报告。

请按照以下格式生成报告：

# [主题] - 技术分析报告

**分析日期**: [当前日期]  
**综合评级**: ⭐⭐⭐⭐⭐ ([评分]/5星)  
**建议**: [总体建议]

## 核心问题识别

[识别和描述核心问题或机会]

## 技术价值量化

### 直接效益
- [列出具体的直接效益]

### 间接效益  
- [列出具体的间接效益]

## 深度技术解析

### 核心优势
[详细分析技术优势]

### 技术限制与风险
[识别技术限制和潜在风险]

## 集成方案设计

### 架构设计
[提供技术架构建议]

### 核心工具集
[列出所需的工具和技术栈]

## 具体使用场景

[描述3-5个具体的使用场景]

## 最佳实践制定

[提供实施的最佳实践]

## 分阶段实施方案

### Phase 1: 概念验证（时间范围）
**目标**: [阶段目标]
**任务**: [具体任务列表]
**成功指标**: [成功标准]

### Phase 2: [下一阶段名称]（时间范围）
**目标**: [阶段目标]  
**任务**: [具体任务列表]
**成功指标**: [成功标准]

### Phase 3: [最终阶段名称]（时间范围）
**目标**: [阶段目标]
**任务**: [具体任务列表]  
**成功指标**: [成功标准]

## 最终建议

**[推荐/不推荐]实施此方案**，理由如下：

1. **[方面1]**: [理由]
2. **[方面2]**: [理由]  
3. **[方面3]**: [理由]

**实施建议**：
- [具体建议1]
- [具体建议2]
- [具体建议3]

请基于内容的技术特性、实施可行性、潜在价值和风险评估进行全面分析。
EOF
}

# 生成简化分析（用于快速评估）
generate_quick_analysis() {
    local content="$1"
    
    local quick_prompt="请对以下内容进行快速分析，重点关注：
1. 核心价值和机会
2. 主要风险和挑战  
3. 是否值得深入研究
4. 3-5个关键行动建议

请用简洁的要点形式回答，不超过500字。"
    
    call_openrouter_api "$content" "$quick_prompt"
}

# 生成深度分析报告
generate_deep_analysis() {
    local content="$1"
    local analysis_prompt=$(build_analysis_prompt)
    
    call_openrouter_api "$content" "$analysis_prompt"
}

# 从inbox.md中移除已处理的项目
remove_from_inbox() {
    local content="$1"
    local inbox_file="$PROJECT_ROOT/inbox.md"
    
    # 创建临时文件，排除包含指定内容的行
    grep -v -F "$content" "$inbox_file" > "${inbox_file}.tmp" || true
    mv "${inbox_file}.tmp" "$inbox_file"
}

# 显示帮助
show_help() {
    cat << EOF
GalaxyAI Content Analysis Script

用法:
  $0 [选项] "要分析的内容"
  
选项:
  -h, --help      显示此帮助信息
  -q, --quick     快速分析模式（简化版）
  -f, --file      从文件读取内容进行分析
  --no-remove     不从inbox.md中移除已处理的项目

示例:
  $0 "git worktree工作流优化方案"
  $0 -q "新的AI辅助编程工具"
  $0 -f /path/to/content.txt
  
环境变量:
  OPENROUTER_API_KEY    OpenRouter API密钥（必需）
  OPENROUTER_MODEL      使用的模型（默认：anthropic/claude-3.5-sonnet）

EOF
}

# 交互式选择模式
interactive_select() {
    local inbox_file="$PROJECT_ROOT/inbox.md"
    
    if [[ ! -f "$inbox_file" ]]; then
        echo "❌ inbox.md 文件不存在"
        exit 1
    fi
    
    echo "📋 从收件箱中选择要分析的内容:"
    echo ""
    
    # 提取待处理项目
    local items=()
    local counter=1
    
    while IFS= read -r line; do
        if [[ "$line" =~ ^-\ \[.*\]\ \[.*\] ]]; then
            echo "$counter) $line"
            items+=("$line")
            ((counter++))
        fi
    done < <(sed -n '/^## 待处理项目/,/^##/p' "$inbox_file" | grep -E '^-\ \[.*\]')
    
    if [[ ${#items[@]} -eq 0 ]]; then
        echo "📭 收件箱为空，没有待处理的项目"
        exit 0
    fi
    
    echo ""
    read -p "请选择要分析的项目编号 (1-${#items[@]}): " selection
    
    if [[ "$selection" =~ ^[0-9]+$ ]] && [[ "$selection" -ge 1 ]] && [[ "$selection" -le ${#items[@]} ]]; then
        echo "${items[$((selection-1))]}"
    else
        echo "❌ 无效的选择"
        exit 1
    fi
}

# 主程序
main() {
    local quick_mode=false
    local from_file=false
    local no_remove=false
    local content=""
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -q|--quick)
                quick_mode=true
                shift
                ;;
            -f|--file)
                from_file=true
                shift
                ;;
            --no-remove)
                no_remove=true
                shift
                ;;
            -*)
                echo "未知选项: $1"
                echo "使用 $0 --help 查看帮助"
                exit 1
                ;;
            *)
                content="$1"
                shift
                ;;
        esac
    done
    
    # 检查配置
    check_config
    
    # 确保分析目录存在
    mkdir -p "$ANALYSIS_DIR"
    
    # 获取要分析的内容
    if [[ -z "$content" ]]; then
        if [[ "$from_file" == true ]]; then
            echo "请指定要读取的文件路径"
            exit 1
        else
            # 交互式选择
            content=$(interactive_select)
        fi
    elif [[ "$from_file" == true ]]; then
        if [[ ! -f "$content" ]]; then
            echo "❌ 文件不存在: $content"
            exit 1
        fi
        content=$(cat "$content")
    fi
    
    if [[ -z "$content" ]]; then
        echo "❌ 没有内容需要分析"
        exit 1
    fi
    
    echo "📝 开始分析内容:"
    echo "$(echo "$content" | head -c 100)..."
    echo ""
    
    # 生成分析
    local analysis=""
    if [[ "$quick_mode" == true ]]; then
        echo "🚀 使用快速分析模式..."
        analysis=$(generate_quick_analysis "$content")
        local filename="quick-analysis-$(date +%Y%m%d_%H%M%S).md"
    else
        echo "🔬 使用深度分析模式..."
        analysis=$(generate_deep_analysis "$content")
        local filename=$(generate_filename "$content")
    fi
    
    # 保存分析结果
    local output_file="$ANALYSIS_DIR/$filename"
    echo "$analysis" > "$output_file"
    
    echo ""
    echo "✅ 分析完成！"
    echo "📄 报告已保存: $output_file"
    
    # 从inbox中移除已处理的项目（如果不是从文件读取且未禁用移除）
    if [[ "$from_file" == false && "$no_remove" == false ]]; then
        remove_from_inbox "$content"
        echo "🗑️  已从收件箱中移除处理项目"
    fi
    
    # 显示通知
    osascript -e "display notification \"分析报告已生成: $filename\" with title \"GalaxyAI Analysis\""
    
    # 询问是否打开报告
    read -p "是否立即查看分析报告? (y/N): " open_report
    if [[ "$open_report" =~ ^[Yy]$ ]]; then
        open "$output_file"
    fi
}

main "$@"