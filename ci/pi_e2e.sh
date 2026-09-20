#!/usr/bin/env bash
# ci/pi_e2e.sh — 真实 Pi harness + 免费模型 + Progress Guard 链路的 CI 冒烟评测
#
# 前置：环境变量
#   LLM_BASE_URL   OpenAI 兼容服务地址（如 https://api.groq.com/openai/v1）
#   LLM_MODEL      模型 id
#   LLM_API_KEY    密钥
# 流程：封闭 Pi home -> 挂 progress-guard 扩展 -> 骨架守卫服务(--lazy，无需权重)
#       -> 跑一个小任务 -> 断言守卫服务收到真实裁决（stdout 遥测）。
set -euo pipefail

if [ -z "${LLM_BASE_URL:-}" ] || [ -z "${LLM_MODEL:-}" ] || [ -z "${LLM_API_KEY:-}" ]; then
  echo "SKIP: 免费模型未配置（设置 secrets.LLM_API_KEY + vars.LLM_BASE_URL + vars.LLM_MODEL 后手动重新触发本 job）"
  exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="$(mktemp -d)"
HOME_PI="$WORK/pi-home"
mkdir -p "$HOME_PI" "$WORK/task"

# 1) 封闭 Pi home：免费模型 provider（OpenAI 兼容）
cat > "$HOME_PI/models.json" <<EOF
{
  "providers": {
    "free-llm": {
      "baseUrl": "$LLM_BASE_URL",
      "api": "openai-completions",
      "apiKey": "$LLM_API_KEY",
      "models": [
        { "id": "$LLM_MODEL", "name": "free-tier model", "reasoning": false, "input": ["text"] }
      ]
    }
  }
}
EOF
cat > "$HOME_PI/settings.json" <<'EOF'
{ "theme": "dark", "autoUpdate": false }
EOF

# 2) 骨架守卫服务（无需权重）
moonbow guard serve --lazy > "$WORK/guard-service.log" 2>&1 &
GUARD_PID=$!
trap 'kill $GUARD_PID 2>/dev/null || true' EXIT
for i in $(seq 1 30); do
  curl -s --noproxy '*' --max-time 2 http://127.0.0.1:18492/health | grep -q ok && break
  sleep 1
done
curl -s --noproxy '*' http://127.0.0.1:18492/health | grep -q ok || { echo "guard service 未就绪"; cat "$WORK/guard-service.log"; exit 1; }

# 3) 小任务：真实 Pi + 守卫扩展
cd "$WORK/task"
git init -q . && echo "# task" > README.md
export PI_CODING_AGENT_DIR="$HOME_PI"
export MOONBOW_GUARD_URL="http://127.0.0.1:18492"

set +e
pi \
  --provider free-llm \
  --model "$LLM_MODEL" \
  --print \
  --no-extensions \
  -e "$REPO/src/moonbow/guard/extensions/progress-guard.ts" \
  --no-skills --no-themes --no-prompt-templates --no-context-files \
  --no-session \
  "请创建文件 greeting.txt，内容为一行：hello moonbow。完成后直接结束。" \
  > "$WORK/pi-stdout.log" 2>&1
PI_RC=$?
set -e

echo "pi exit=$PI_RC"
tail -c 800 "$WORK/pi-stdout.log" || true

# 4) 断言：守卫服务收到了真实 /check 裁决
if grep -q "/check verdict=" "$WORK/guard-service.log"; then
  echo "PASS: guard service 收到真实裁决"
  grep "/check verdict=" "$WORK/guard-service.log"
  exit 0
else
  echo "FAIL: 服务日志中没有 /check 裁决（守卫链路未接通）"
  echo "--- guard-service.log ---"; tail -20 "$WORK/guard-service.log"
  exit 1
fi
