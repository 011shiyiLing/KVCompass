#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_instances.sh — launch the 3-region testbed with the AWS CLI.
#
# Prereqs: `aws configure` done (IAM user with EC2 full access), quotas
# granted, and an SSH public key at
# $PUBKEY (default ~/.ssh/id_ed25519.pub — imported into every region).
#
#   bash scripts/aws/launch_instances.sh            # launch everything
#   SPOT=1 bash scripts/aws/launch_instances.sh     # GPU nodes as spot
#   bash scripts/aws/launch_instances.sh ips        # re-print hosts.env block
#   bash scripts/aws/launch_instances.sh authorize  # open 8001/30000 between nodes
#   bash scripts/aws/launch_instances.sh terminate  # kill EVERYTHING tagged
#
# All instances are tagged Project=kvcompass so `terminate` can't touch
# anything else in the account.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REGIONS=("us-east-1" "eu-central-1" "ap-northeast-1")
GPU_COUNTS=(6 2 4)                       # census-proportional US/EU/Asia
GPU_TYPE="g6.xlarge"                     # 1×L4 24GB
CPU_TYPE="c6i.2xlarge"                   # router + replayer node
KEY_NAME="${KEY_NAME:-kvcompass}"
PUBKEY="${PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
SG_NAME="kvcompass-sg"
TAG="Project=kvcompass"
SPOT="${SPOT:-0}"

# SSM public parameters resolve the right AMI per region automatically.
GPU_AMI_PARAM="/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
CPU_AMI_PARAM="/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id"

cmd="${1:-launch}"

my_ip() { curl -s https://checkip.amazonaws.com; }

ensure_key_and_sg() {   # $1 = region
  local r="$1"
  aws ec2 describe-key-pairs --region "$r" --key-names "$KEY_NAME" \
    >/dev/null 2>&1 || \
    aws ec2 import-key-pair --region "$r" --key-name "$KEY_NAME" \
      --public-key-material "fileb://$PUBKEY" >/dev/null
  local sg
  sg=$(aws ec2 describe-security-groups --region "$r" \
        --filters "Name=group-name,Values=$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
  if [[ "$sg" == "None" || -z "$sg" ]]; then
    sg=$(aws ec2 create-security-group --region "$r" \
          --group-name "$SG_NAME" --description "kvcompass 3-region testbed" \
          --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --region "$r" --group-id "$sg" \
      --protocol tcp --port 22 --cidr "$(my_ip)/32" >/dev/null
  fi
  echo "$sg"
}

launch_one_region() {   # $1=region  $2=gpu_count
  local r="$1" n_gpu="$2" sg gpu_ami cpu_ami market=()
  sg=$(ensure_key_and_sg "$r")
  gpu_ami=$(aws ssm get-parameter --region "$r" --name "$GPU_AMI_PARAM" \
             --query 'Parameter.Value' --output text)
  cpu_ami=$(aws ssm get-parameter --region "$r" --name "$CPU_AMI_PARAM" \
             --query 'Parameter.Value' --output text)
  [[ "$SPOT" == "1" ]] && market=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}')

  echo "[$r] launching $n_gpu × $GPU_TYPE (workers) ..."
  aws ec2 run-instances --region "$r" \
    --image-id "$gpu_ami" --instance-type "$GPU_TYPE" --count "$n_gpu" \
    --key-name "$KEY_NAME" --security-group-ids "$sg" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    "${market[@]}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=kvcompass},{Key=Role,Value=worker}]" \
    --query 'Instances[].InstanceId' --output text

  echo "[$r] launching 1 × $CPU_TYPE (router) ..."
  aws ec2 run-instances --region "$r" \
    --image-id "$cpu_ami" --instance-type "$CPU_TYPE" --count 1 \
    --key-name "$KEY_NAME" --security-group-ids "$sg" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=kvcompass},{Key=Role,Value=router}]" \
    --query 'Instances[].InstanceId' --output text
}

list_ips() {            # $1=region $2=role
  aws ec2 describe-instances --region "$1" \
    --filters "Name=tag:Project,Values=kvcompass" "Name=tag:Role,Values=$2" \
              "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[].Instances[].PublicIpAddress' --output text
}

print_hosts_env() {
  echo "# ── paste into scripts/aws/hosts.env ──"
  local routers=()
  for r in "${REGIONS[@]}"; do
    routers+=("$(list_ips "$r" router)")
  done
  echo "export ROUTER_HOSTS=($(printf '"%s" ' "${routers[@]}"))"
  local i=0
  for r in "${REGIONS[@]}"; do
    # shellcheck disable=SC2207
    local ws=($(list_ips "$r" worker))
    echo "export WORKER_HOSTS_${i}=($(printf '"%s" ' "${ws[@]}"))   # $r ×${#ws[@]}"
    i=$((i+1))
  done
}

authorize_mesh() {
  # Open router (8001) + SGLang (30000) ports between every testbed IP.
  local all_ips=()
  for r in "${REGIONS[@]}"; do
    # shellcheck disable=SC2207
    all_ips+=($(list_ips "$r" worker) $(list_ips "$r" router))
  done
  echo "authorizing 8001/30000 between ${#all_ips[@]} node IPs in all regions ..."
  for r in "${REGIONS[@]}"; do
    local sg
    sg=$(aws ec2 describe-security-groups --region "$r" \
          --filters "Name=group-name,Values=$SG_NAME" \
          --query 'SecurityGroups[0].GroupId' --output text)
    for ip in "${all_ips[@]}"; do
      for port in 8001 30000; do
        aws ec2 authorize-security-group-ingress --region "$r" \
          --group-id "$sg" --protocol tcp --port "$port" \
          --cidr "$ip/32" >/dev/null 2>&1 || true   # idempotent
      done
    done
  done
  echo "done"
}

case "$cmd" in
  launch)
    for i in "${!REGIONS[@]}"; do
      launch_one_region "${REGIONS[$i]}" "${GPU_COUNTS[$i]}"
    done
    echo "waiting 60s for public IPs ..."; sleep 60
    print_hosts_env
    authorize_mesh
    ;;
  ips)        print_hosts_env ;;
  authorize)  authorize_mesh ;;
  terminate)
    for r in "${REGIONS[@]}"; do
      ids=$(aws ec2 describe-instances --region "$r" \
        --filters "Name=tag:Project,Values=kvcompass" \
                  "Name=instance-state-name,Values=running,pending,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text)
      if [[ -n "$ids" ]]; then
        echo "[$r] terminating: $ids"
        # shellcheck disable=SC2086
        aws ec2 terminate-instances --region "$r" --instance-ids $ids \
          --query 'TerminatingInstances[].InstanceId' --output text
      fi
    done
    ;;
  *) echo "usage: $0 launch|ips|authorize|terminate"; exit 1 ;;
esac
