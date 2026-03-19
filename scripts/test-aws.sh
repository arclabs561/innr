#!/usr/bin/env bash
# Launch EC2 spot instances for cross-architecture testing and benchmarking.
#
# Usage:
#   ./scripts/test-aws.sh [--bench] [--arch x86|arm|avx512|all]
#
# Instance types:
#   x86:    c5.xlarge    (Skylake, AVX2+FMA)
#   avx512: c6i.xlarge   (Ice Lake, AVX-512)
#   arm:    c7g.xlarge   (Graviton3, NEON+SVE)
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - An SSH key pair registered in EC2 (set KEY_NAME below or via env)
#   - A security group allowing SSH inbound (set SG_ID below or via env)
#
# The script:
#   1. Launches a spot instance with the right AMI for each arch
#   2. Waits for SSH to become available
#   3. Installs Rust, runs tests/benchmarks via test-remote.sh
#   4. Downloads results and terminates the instance

set -euo pipefail

# Configuration (override via environment)
KEY_NAME="${AWS_KEY_NAME:-innr-test}"
KEY_FILE="${AWS_KEY_FILE:-~/.ssh/${KEY_NAME}.pem}"
SG_ID="${AWS_SG_ID:-}"
SUBNET_ID="${AWS_SUBNET_ID:-}"
REGION="${AWS_REGION:-us-east-1}"

RUN_BENCH=false
ARCH="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench) RUN_BENCH=true; shift ;;
        --arch)  ARCH="$2"; shift 2 ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# AMI IDs: Amazon Linux 2023 (latest, free tier eligible)
# These are region-specific; using us-east-1 defaults
get_ami() {
    local arch_filter="$1"
    aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=al2023-ami-2023*-kernel-*-${arch_filter}" \
            "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text
}

# Instance configurations
declare -A INSTANCE_TYPES=(
    [x86]="c5.xlarge"
    [avx512]="c6i.xlarge"
    [arm]="c7g.xlarge"
)

declare -A ARCH_FILTERS=(
    [x86]="x86_64"
    [avx512]="x86_64"
    [arm]="arm64"
)

declare -A SSH_USERS=(
    [x86]="ec2-user"
    [avx512]="ec2-user"
    [arm]="ec2-user"
)

declare -A DESCRIPTIONS=(
    [x86]="Skylake (AVX2+FMA)"
    [avx512]="Ice Lake (AVX-512)"
    [arm]="Graviton3 (NEON)"
)

launch_and_test() {
    local arch="$1"
    local instance_type="${INSTANCE_TYPES[$arch]}"
    local ami_arch="${ARCH_FILTERS[$arch]}"
    local ssh_user="${SSH_USERS[$arch]}"
    local desc="${DESCRIPTIONS[$arch]}"

    echo ""
    echo "========================================"
    echo "  $arch: $desc ($instance_type)"
    echo "========================================"
    echo ""

    # Get latest AMI
    echo "Looking up AMI for $ami_arch..."
    local ami_id
    ami_id=$(get_ami "$ami_arch")
    if [[ "$ami_id" == "None" || -z "$ami_id" ]]; then
        echo "ERROR: Could not find AMI for $ami_arch in $REGION"
        return 1
    fi
    echo "AMI: $ami_id"

    # User data: install Rust on boot
    local user_data
    user_data=$(base64 <<'USERDATA'
#!/bin/bash
yum install -y gcc git
su - ec2-user -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
touch /tmp/rust-ready
USERDATA
)

    # Build launch args
    local launch_args=(
        --region "$REGION"
        --image-id "$ami_id"
        --instance-type "$instance_type"
        --key-name "$KEY_NAME"
        --user-data "$user_data"
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}'
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=innr-test-${arch}},{Key=Project,Value=innr}]"
        --query 'Instances[0].InstanceId'
        --output text
    )

    if [[ -n "$SG_ID" ]]; then
        launch_args+=(--security-group-ids "$SG_ID")
    fi
    if [[ -n "$SUBNET_ID" ]]; then
        launch_args+=(--subnet-id "$SUBNET_ID")
    fi

    echo "Launching spot instance..."
    local instance_id
    instance_id=$(aws ec2 run-instances "${launch_args[@]}")
    echo "Instance: $instance_id"

    # Cleanup trap
    trap "echo 'Terminating $instance_id...'; aws ec2 terminate-instances --region $REGION --instance-ids $instance_id > /dev/null 2>&1" EXIT

    # Wait for running
    echo "Waiting for instance to start..."
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$instance_id"

    # Get public IP
    local public_ip
    public_ip=$(aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    echo "IP: $public_ip"

    # Wait for SSH
    echo "Waiting for SSH..."
    local ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o UserKnownHostsFile=/dev/null -i $KEY_FILE"
    for i in $(seq 1 30); do
        if ssh $ssh_opts "$ssh_user@$public_ip" "true" 2>/dev/null; then
            break
        fi
        if [[ $i -eq 30 ]]; then
            echo "ERROR: SSH timeout after 150s"
            return 1
        fi
        sleep 5
    done

    # Wait for Rust installation (user-data)
    echo "Waiting for Rust installation..."
    for i in $(seq 1 60); do
        if ssh $ssh_opts "$ssh_user@$public_ip" "test -f /tmp/rust-ready" 2>/dev/null; then
            break
        fi
        if [[ $i -eq 60 ]]; then
            echo "ERROR: Rust install timeout after 300s"
            return 1
        fi
        sleep 5
    done

    echo "Running tests..."
    local bench_flag=""
    if [[ "$RUN_BENCH" == "true" ]]; then
        bench_flag="--bench"
    fi

    # Use test-remote.sh with SSH options baked in
    SSH_TARGET="$ssh_user@$public_ip"
    export RSYNC_RSH="ssh $ssh_opts"
    "$SCRIPT_DIR/test-remote.sh" "$SSH_TARGET" $bench_flag 2>&1 | \
        tee "bench_results/remote/${arch}.log"

    echo ""
    echo "$arch: DONE"

    # Terminate
    echo "Terminating $instance_id..."
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$instance_id" > /dev/null
    trap - EXIT
}

# Main
echo "=== innr AWS Cross-Architecture Testing ==="
echo "Region: $REGION"
echo "Key: $KEY_NAME"
echo "Bench: $RUN_BENCH"
echo ""

mkdir -p bench_results/remote

case "$ARCH" in
    all)
        for arch in x86 avx512 arm; do
            launch_and_test "$arch" || echo "WARN: $arch failed"
        done
        ;;
    x86|avx512|arm)
        launch_and_test "$ARCH"
        ;;
    *)
        echo "Unknown arch: $ARCH (choose: x86, avx512, arm, all)"
        exit 1
        ;;
esac

echo ""
echo "=== All done ==="
echo "Results in bench_results/remote/"
