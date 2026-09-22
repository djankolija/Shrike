# The mini's production server launch, written once and sourced by
# tools/mini-deploy.sh (--restart) and both rigs (their restore, and the arm every
# A/B layers onto); CLAUDE.md's launch line is a copy of this one. The slot table
# is ornith15's at 160 slots per layer (v22 Task 3) and must sum to what the budget
# snaps to, or the launch is refused.
PRODUCTION_ENV="SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1 SHRIKE_EXPERT_SLOT_TABLE=256,256,246,209,191,171,171,162,171,149,164,169,155,144,137,135,133,131,132,130,145,133,141,142,137,137,130,137,137,142,137,135,157,157,162,160,166,161,178,194 SHRIKE_EXPERT_POLICY=slru"
PRODUCTION_MODEL=./models/ornith15.gturbo
PRODUCTION_PORT=8081
PRODUCTION_RAM_BUDGET=11324620800
SERVER_LOG=/tmp/shrike-server.log

# server_launch <model> <port> <ram budget> [assignment ...]: the command, run from
# ~/shrike-runtime, that starts the server in the background. Extra assignments
# follow production's, so an arm overrides a production value by assigning it again
# (env keeps the last); an empty SHRIKE_EXPERT_SLOT_TABLE= runs the uniform pool.
server_launch() {
  local model=$1 port=$2 budget=$3
  shift 3
  echo "env $PRODUCTION_ENV $* nohup ./bin/shrike serve --model $model --port $port --max-context 32768 --ram-budget $budget --thinking off > $SERVER_LOG 2>&1 &"
}
