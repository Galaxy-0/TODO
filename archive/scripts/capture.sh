#!/bin/bash

# GalaxyAI Content Capture Script
# 用于从任意macOS应用程序捕获内容到inbox.md

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INBOX_FILE="$PROJECT_ROOT/inbox.md"

# 确保inbox.md存在
if [[ ! -f "$INBOX_FILE" ]]; then
    echo "错误: inbox.md 文件不存在: $INBOX_FILE"
    exit 1
fi

# 获取当前时间戳
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# 获取当前活跃应用
get_active_app() {
    osascript <<EOF
tell application "System Events"
    set activeApp to name of first application process whose frontmost is true
    return activeApp
end tell
EOF
}

# 从浏览器获取内容 (Safari, Chrome, Arc等)
capture_browser_content() {
    local app_name="$1"
    
    case "$app_name" in
        "Safari" | "Safari Technology Preview")
            osascript <<EOF
tell application "$app_name"
    set currentTab to current tab of front window
    set pageURL to URL of currentTab
    set pageTitle to name of currentTab
    set selectedText to ""
    
    try
        # 尝试获取选中文本
        tell application "System Events"
            keystroke "c" using command down
            delay 0.1
        end tell
        set selectedText to (the clipboard as string)
    on error
        set selectedText to ""
    end try
    
    return pageURL & "|" & pageTitle & "|" & selectedText
end tell
EOF
            ;;
        "Google Chrome" | "Chromium" | "Arc")
            osascript <<EOF
tell application "$app_name"
    set currentTab to active tab of front window
    set pageURL to URL of currentTab
    set pageTitle to title of currentTab
    set selectedText to ""
    
    try
        tell application "System Events"
            keystroke "c" using command down
            delay 0.1
        end tell
        set selectedText to (the clipboard as string)
    on error
        set selectedText to ""
    end try
    
    return pageURL & "|" & pageTitle & "|" & selectedText
end tell
EOF
            ;;
        *)
            echo "|||"
            ;;
    esac
}

# 从任意应用获取选中文本
capture_selected_text() {
    osascript <<EOF
tell application "System Events"
    keystroke "c" using command down
    delay 0.1
end tell

try
    return (the clipboard as string)
on error
    return ""
end try
EOF
}

# 获取窗口标题
get_window_title() {
    osascript <<EOF
tell application "System Events"
    set activeApp to name of first application process whose frontmost is true
    tell process activeApp
        try
            set windowTitle to name of front window
            return windowTitle
        on error
            return ""
        end try
    end tell
end tell
EOF
}

# 主捕获逻辑
capture_content() {
    local app_name=$(get_active_app)
    local content=""
    local source=""
    local url=""
    local title=""
    
    echo "检测到活跃应用: $app_name"
    
    case "$app_name" in
        "Safari" | "Safari Technology Preview" | "Google Chrome" | "Chromium" | "Arc")
            echo "从浏览器捕获内容..."
            local browser_data=$(capture_browser_content "$app_name")
            IFS='|' read -r url title content <<< "$browser_data"
            source="浏览器($app_name)"
            ;;
        *)
            echo "从应用程序捕获选中文本..."
            content=$(capture_selected_text)
            title=$(get_window_title)
            source="应用($app_name)"
            ;;
    esac
    
    # 格式化输出内容
    local formatted_content="- [$TIMESTAMP] [$source]"
    
    if [[ -n "$title" ]]; then
        formatted_content="$formatted_content **$title**"
    fi
    
    if [[ -n "$url" ]]; then
        formatted_content="$formatted_content\n  - URL: $url"
    fi
    
    if [[ -n "$content" && ${#content} -gt 5 ]]; then
        # 限制内容长度，避免过长
        if [[ ${#content} -gt 500 ]]; then
            content="${content:0:500}..."
        fi
        formatted_content="$formatted_content\n  - 内容: $content"
    fi
    
    # 添加到inbox.md
    echo "将内容添加到 $INBOX_FILE"
    
    # 在"待处理项目"标题后插入新内容
    awk -v new_content="$formatted_content" '
        /^## 待处理项目/ {
            print $0
            print ""
            print new_content
            print ""
            next
        }
        { print }
    ' "$INBOX_FILE" > "${INBOX_FILE}.tmp" && mv "${INBOX_FILE}.tmp" "$INBOX_FILE"
    
    echo "✅ 内容已成功捕获到inbox.md"
    
    # 可选：显示通知
    osascript -e "display notification \"内容已捕获到GalaxyAI收件箱\" with title \"GalaxyAI\""
}

# 截图捕获功能
capture_screenshot() {
    local screenshot_dir="$PROJECT_ROOT/screenshots"
    mkdir -p "$screenshot_dir"
    
    local filename="screenshot_$(date +%Y%m%d_%H%M%S).png"
    local filepath="$screenshot_dir/$filename"
    
    echo "开始截图..."
    screencapture -i "$filepath"
    
    if [[ -f "$filepath" ]]; then
        echo "截图已保存: $filepath"
        
        # 添加到inbox.md
        local formatted_content="- [$TIMESTAMP] [截图] **$filename**\n  - 路径: $filepath"
        
        awk -v new_content="$formatted_content" '
            /^## 待处理项目/ {
                print $0
                print ""
                print new_content
                print ""
                next
            }
            { print }
        ' "$INBOX_FILE" > "${INBOX_FILE}.tmp" && mv "${INBOX_FILE}.tmp" "$INBOX_FILE"
        
        echo "✅ 截图信息已添加到inbox.md"
        osascript -e "display notification \"截图已捕获到GalaxyAI收件箱\" with title \"GalaxyAI\""
    else
        echo "❌ 截图取消或失败"
    fi
}

# 手动输入捕获
capture_manual_input() {
    echo "请输入要捕获的内容 (按Ctrl+D结束):"
    local content=$(cat)
    
    if [[ -n "$content" ]]; then
        local formatted_content="- [$TIMESTAMP] [手动输入] $content"
        
        awk -v new_content="$formatted_content" '
            /^## 待处理项目/ {
                print $0
                print ""
                print new_content
                print ""
                next
            }
            { print }
        ' "$INBOX_FILE" > "${INBOX_FILE}.tmp" && mv "${INBOX_FILE}.tmp" "$INBOX_FILE"
        
        echo "✅ 手动输入已捕获到inbox.md"
    else
        echo "❌ 没有输入内容"
    fi
}

# 显示帮助
show_help() {
    cat << EOF
GalaxyAI Content Capture Script

用法:
  $0 [选项]

选项:
  -h, --help      显示此帮助信息
  -s, --screenshot 截图模式
  -m, --manual    手动输入模式
  (无参数)         自动检测当前应用并捕获内容

示例:
  $0              # 捕获当前应用的内容
  $0 -s           # 进入截图模式
  $0 -m           # 手动输入内容

EOF
}

# 主程序
main() {
    case "${1:-}" in
        -h|--help)
            show_help
            ;;
        -s|--screenshot)
            capture_screenshot
            ;;
        -m|--manual)
            capture_manual_input
            ;;
        "")
            capture_content
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 $0 --help 查看帮助"
            exit 1
            ;;
    esac
}

main "$@"