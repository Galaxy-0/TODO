#!/bin/bash

# GalaxyAI Setup Script
# 用于初始化和配置GalaxyAI系统

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🌟 欢迎使用 GalaxyAI 个人助理系统！"
echo ""

# 检查系统要求
check_requirements() {
    echo "🔍 检查系统要求..."
    
    # 检查macOS版本
    if [[ "$(uname)" != "Darwin" ]]; then
        echo "❌ 此系统仅支持 macOS"
        exit 1
    fi
    
    # 检查必需的命令
    local required_commands=("curl" "jq" "osascript")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "❌ 缺少必需的命令: $cmd"
            
            case "$cmd" in
                "jq")
                    echo "请运行: brew install jq"
                    ;;
                *)
                    echo "请安装 $cmd"
                    ;;
            esac
            exit 1
        fi
    done
    
    echo "✅ 系统要求检查通过"
}

# 安装可选依赖
install_optional_deps() {
    echo ""
    echo "📦 检查可选依赖..."
    
    # 检查Homebrew
    if ! command -v brew &> /dev/null; then
        echo "⚠️  未检测到 Homebrew，建议安装以获得更好的体验"
        read -p "是否安装 Homebrew? (y/N): " install_brew
        if [[ "$install_brew" =~ ^[Yy]$ ]]; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
    fi
    
    # 检查tesseract（OCR功能）
    if ! command -v tesseract &> /dev/null; then
        echo "⚠️  未检测到 tesseract（OCR功能将不可用）"
        read -p "是否安装 tesseract 以启用OCR功能? (y/N): " install_tesseract
        if [[ "$install_tesseract" =~ ^[Yy]$ ]]; then
            if command -v brew &> /dev/null; then
                brew install tesseract tesseract-lang
                echo "✅ tesseract 安装完成"
            else
                echo "❌ 需要 Homebrew 来安装 tesseract"
            fi
        fi
    else
        echo "✅ tesseract 已安装"
    fi
}

# 配置API密钥
setup_api_key() {
    echo ""
    echo "🔑 配置 OpenRouter API 密钥"
    echo ""
    echo "GalaxyAI 使用 OpenRouter 来调用多种AI模型进行分析。"
    echo "请访问 https://openrouter.ai 注册账户并获取API密钥。"
    echo ""
    
    local config_file="config/config.env"
    
    if [[ -f "$config_file" ]]; then
        local current_key=$(grep "OPENROUTER_API_KEY=" "$config_file" | cut -d'=' -f2)
        if [[ -n "$current_key" && "$current_key" != " " ]]; then
            echo "✅ 检测到已配置的API密钥"
            read -p "是否更新API密钥? (y/N): " update_key
            if [[ ! "$update_key" =~ ^[Yy]$ ]]; then
                return
            fi
        fi
    fi
    
    read -p "请输入您的 OpenRouter API 密钥: " api_key
    
    if [[ -z "$api_key" ]]; then
        echo "⚠️  跳过API密钥配置（稍后可手动编辑 config/config.env）"
        return
    fi
    
    # 更新配置文件
    if [[ -f "$config_file" ]]; then
        sed -i.bak "s/OPENROUTER_API_KEY=.*/OPENROUTER_API_KEY=$api_key/" "$config_file"
        rm -f "${config_file}.bak"
    else
        echo "❌ 配置文件不存在: $config_file"
        return
    fi
    
    echo "✅ API密钥配置完成"
}

# 设置macOS权限
setup_permissions() {
    echo ""
    echo "🔐 配置 macOS 权限"
    echo ""
    echo "GalaxyAI 需要以下权限来正常工作："
    echo "- 辅助功能权限（用于读取其他应用程序的内容）"
    echo "- 屏幕录制权限（用于截图功能）"
    echo ""
    
    read -p "是否现在打开系统偏好设置来配置权限? (y/N): " setup_perms
    if [[ "$setup_perms" =~ ^[Yy]$ ]]; then
        echo "正在打开系统偏好设置..."
        open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        echo ""
        echo "请在「隐私与安全性」->「辅助功能」中添加以下应用："
        echo "- Terminal（或您使用的终端应用）"
        echo "- 任何您想要运行脚本的应用"
        echo ""
        read -p "配置完成后按回车继续..."
        
        echo "正在打开屏幕录制权限设置..."
        open "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
        echo ""
        echo "请在「隐私与安全性」->「屏幕录制」中添加相同的应用"
        echo ""
        read -p "配置完成后按回车继续..."
    fi
}

# 创建快捷键设置
setup_shortcuts() {
    echo ""
    echo "⌨️  设置全局快捷键"
    echo ""
    echo "建议为 capture.sh 脚本创建全局快捷键，以便快速捕获内容。"
    echo ""
    echo "推荐设置："
    echo "1. 打开「系统偏好设置」->「键盘」->「快捷键」->「服务」"
    echo "2. 或者使用 Alfred、Raycast 等工具创建快捷键"
    echo ""
    echo "脚本路径: $PWD/scripts/capture.sh"
    echo ""
    
    read -p "是否现在打开快捷键设置? (y/N): " setup_shortcuts
    if [[ "$setup_shortcuts" =~ ^[Yy]$ ]]; then
        open "x-apple.systempreferences:com.apple.preference.keyboard?Shortcuts"
    fi
}

# 运行测试
run_test() {
    echo ""
    echo "🧪 运行基本测试..."
    
    # 测试捕获脚本
    if [[ -x "scripts/capture.sh" ]]; then
        echo "✅ capture.sh 可执行"
    else
        echo "❌ capture.sh 不可执行"
        chmod +x scripts/capture.sh
        echo "✅ 已修复 capture.sh 权限"
    fi
    
    # 测试处理脚本
    if [[ -x "scripts/process.sh" ]]; then
        echo "✅ process.sh 可执行"
    else
        echo "❌ process.sh 不可执行"
        chmod +x scripts/process.sh
        echo "✅ 已修复 process.sh 权限"
    fi
    
    # 测试配置文件
    if [[ -f "config/config.env" ]]; then
        echo "✅ 配置文件存在"
        
        # 检查API密钥是否已设置
        local api_key=$(grep "OPENROUTER_API_KEY=" config/config.env | cut -d'=' -f2 | xargs)
        if [[ -n "$api_key" && "$api_key" != "" ]]; then
            echo "✅ API密钥已配置"
        else
            echo "⚠️  API密钥未配置"
        fi
    else
        echo "❌ 配置文件缺失"
    fi
    
    echo "✅ 基本测试完成"
}

# 显示完成信息
show_completion() {
    echo ""
    echo "🎉 GalaxyAI 设置完成！"
    echo ""
    echo "📖 快速开始："
    echo "1. 使用全局快捷键或运行 ./scripts/capture.sh 来捕获内容"
    echo "2. 运行 ./scripts/process.sh 来分析捕获的内容"
    echo "3. 查看 analysis/ 目录中生成的分析报告"
    echo ""
    echo "📚 更多信息："
    echo "- 配置文件: config/config.env"
    echo "- 捕获的内容: inbox.md"
    echo "- 分析报告: analysis/"
    echo "- 帮助信息: ./scripts/capture.sh --help"
    echo "            ./scripts/process.sh --help"
    echo ""
    echo "🚀 开始使用 GalaxyAI 提升您的工作效率吧！"
}

# 主程序
main() {
    check_requirements
    install_optional_deps
    setup_api_key
    setup_permissions
    setup_shortcuts
    run_test
    show_completion
}

# 检查是否在交互式终端中运行
if [[ -t 0 ]]; then
    main
else
    echo "请在交互式终端中运行此脚本"
    exit 1
fi