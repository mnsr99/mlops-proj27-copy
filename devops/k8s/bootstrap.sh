#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# bootstrap.sh — one-command cluster setup for Jitsi + Platform (Postgres,
# MinIO, MLflow) on k3s.
#
# Prerequisite: /mnt/block is already mounted (Cinder volume proj27-platform,
# ext4, UUID in /etc/fstab). See devops/README for the one-time volume setup.
#
# Usage:
#   bash bootstrap.sh <FLOATING_IP> <JICOFO_PASSWORD> <JVB_PASSWORD>
#
# Example:
#   bash bootstrap.sh 129.114.27.109 myJicofoPass myJvbPass
# ---------------------------------------------------------------------------

FLOATING_IP="${1:?Error: FLOATING_IP is required (arg 1)}"
JICOFO_PASS="${2:?Error: JICOFO_PASSWORD is required (arg 2)}"
JVB_PASS="${3:?Error: JVB_PASSWORD is required (arg 3)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="$SCRIPT_DIR"
KUBECTL="sudo kubectl"
# helm does NOT auto-fallback to /etc/rancher/k3s/k3s.yaml the way k3s kubectl
# does, and `sudo` strips env vars — so we run helm unprivileged with an
# explicit KUBECONFIG (chmod 644'd below) instead of `sudo helm`.
HELM="helm"

NIP_DOMAIN="${FLOATING_IP//./-}.nip.io"

echo "============================================"
echo " Bootstrap: Jitsi + Platform on k3s"
echo " Floating IP  : $FLOATING_IP"
echo " nip.io domain: $NIP_DOMAIN"
echo " Manifests dir: $K8S_DIR"
echo "============================================"

# ------------------------------------------------------------------
# 0. Sanity check — /mnt/block must exist and be a mount point
# ------------------------------------------------------------------
if ! mountpoint -q /mnt/block; then
    echo "ERROR: /mnt/block is not mounted."
    echo "  Attach the Cinder volume and mount it before running bootstrap."
    exit 1
fi
echo "[0/9] /mnt/block is mounted. $(df -h /mnt/block | tail -1 | awk '{print $4" free"}')"

# ------------------------------------------------------------------
# 1. Install k3s (skip if already installed)
# ------------------------------------------------------------------
if ! command -v k3s &> /dev/null; then
    echo "[1/9] Installing k3s ..."
    curl -sfL https://get.k3s.io | sh -
    echo "Waiting for k3s node to register ..."
    until $KUBECTL get nodes 2>/dev/null | grep -q " Ready"; do
        sleep 3
    done
    echo "k3s node is Ready."
else
    echo "[1/9] k3s already installed, skipping."
fi

# ------------------------------------------------------------------
# 2. Install helm (skip if already installed)
# ------------------------------------------------------------------
if ! command -v helm &> /dev/null; then
    echo "[2/9] Installing helm ..."
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
else
    echo "[2/9] helm already installed, skipping."
fi

# Point helm at k3s kubeconfig
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
sudo chmod 644 /etc/rancher/k3s/k3s.yaml || true

# Add helm repos
$HELM repo add bitnami https://charts.bitnami.com/bitnami >/dev/null 2>&1 || true
$HELM repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
$HELM repo update >/dev/null

# ------------------------------------------------------------------
# 3. Reconfigure local-path provisioner -> /mnt/block
# ------------------------------------------------------------------
echo "[3/9] Pointing local-path StorageClass at /mnt/block ..."
$KUBECTL apply -f "$K8S_DIR/storage.yaml"
# Restart local-path-provisioner so it picks up the new config
$KUBECTL -n kube-system rollout restart deployment local-path-provisioner || true

# ------------------------------------------------------------------
# 4. Create namespaces
# ------------------------------------------------------------------
echo "[4/9] Creating namespaces ..."
$KUBECTL create namespace jitsi    --dry-run=client -o yaml | $KUBECTL apply -f -
$KUBECTL create namespace platform --dry-run=client -o yaml | $KUBECTL apply -f -

# ------------------------------------------------------------------
# 5. Create Jitsi secrets + TLS
# ------------------------------------------------------------------
echo "[5/9] Creating Jitsi secrets ..."
$KUBECTL create secret generic jitsi-secrets \
    --from-literal=JICOFO_AUTH_PASSWORD="$JICOFO_PASS" \
    --from-literal=JVB_AUTH_PASSWORD="$JVB_PASS" \
    -n jitsi --dry-run=client -o yaml | $KUBECTL apply -f -

echo "      Generating TLS certificate for $NIP_DOMAIN ..."
TLS_DIR=$(mktemp -d)
openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout "$TLS_DIR/tls.key" \
    -out "$TLS_DIR/tls.crt" \
    -subj "/CN=$NIP_DOMAIN" \
    -addext "subjectAltName=DNS:$NIP_DOMAIN" 2>/dev/null

$KUBECTL create secret tls jitsi-tls \
    --cert="$TLS_DIR/tls.crt" \
    --key="$TLS_DIR/tls.key" \
    -n jitsi --dry-run=client -o yaml | $KUBECTL apply -f -
rm -rf "$TLS_DIR"

# ------------------------------------------------------------------
# 6. Deploy platform services: Postgres + MinIO (via Helm)
# ------------------------------------------------------------------
echo "[6/9] Deploying Postgres ..."
$HELM upgrade --install postgres bitnami/postgresql \
    -n platform \
    -f "$K8S_DIR/postgres/values.yaml" \
    --wait --timeout 10m

echo "      Deploying MinIO ..."
# NOTE: Using official minio/minio image directly (not Bitnami chart) because
# Bitnami removed minio images from Docker Hub after Aug 2025 subscription
# change. See minio/minio.yaml for full manifest.
$KUBECTL apply -f "$K8S_DIR/minio/minio.yaml"
echo "      Waiting for MinIO deployment ..."
until $KUBECTL -n platform get deployment minio 2>/dev/null | grep -q "1/1"; do
    sleep 3
done
echo "      Waiting for bucket-creation job ..."
$KUBECTL -n platform wait --for=condition=complete --timeout=5m job/minio-create-buckets || true

# ------------------------------------------------------------------
# 7. Deploy MLflow (depends on Postgres + MinIO)
# ------------------------------------------------------------------
echo "[7/10] Deploying MLflow ..."
$KUBECTL apply -f "$K8S_DIR/mlflow/mlflow.yaml"

# ------------------------------------------------------------------
# 8. Deploy monitoring (Prometheus + Grafana)
# ------------------------------------------------------------------
echo "[8/10] Deploying Prometheus + Grafana ..."
$KUBECTL create namespace monitoring --dry-run=client -o yaml | $KUBECTL apply -f -
$HELM upgrade --install monitoring prometheus-community/kube-prometheus-stack \
    -n monitoring \
    -f "$K8S_DIR/monitoring/values.yaml" \
    --wait --timeout 10m

# ------------------------------------------------------------------
# 9. Deploy Jitsi (substitute placeholders then apply)
# ------------------------------------------------------------------
echo "[9/10] Deploying Jitsi ..."

PROSODY_CLUSTER_IP=""

apply_jitsi_manifest() {
    local file="$1"
    sed \
        -e "s|A-B-C-D.nip.io|${NIP_DOMAIN}|g" \
        -e "s|https://A-B-C-D.nip.io|https://${NIP_DOMAIN}|g" \
        -e "s|<FLOATING_IP>|${FLOATING_IP}|g" \
        -e "s|<PROSODY_CLUSTER_IP>|${PROSODY_CLUSTER_IP}|g" \
        "$file" | $KUBECTL apply -f -
}

apply_jitsi_manifest "$K8S_DIR/jitsi/configmap.yaml"

apply_jitsi_manifest "$K8S_DIR/jitsi/prosody.yaml"
echo "      Waiting for prosody pod to be ready ..."
until $KUBECTL get deployment prosody -n jitsi 2>/dev/null | grep -q "1/1"; do
    sleep 3
done

PROSODY_CLUSTER_IP=$($KUBECTL get svc prosody -n jitsi -o jsonpath='{.spec.clusterIP}')
echo "      Prosody ClusterIP: $PROSODY_CLUSTER_IP"

apply_jitsi_manifest "$K8S_DIR/jitsi/jicofo.yaml"
apply_jitsi_manifest "$K8S_DIR/jitsi/jvb.yaml"
apply_jitsi_manifest "$K8S_DIR/jitsi/web.yaml"
apply_jitsi_manifest "$K8S_DIR/jitsi/ingress.yaml"

# ------------------------------------------------------------------
# 10. Verify
# ------------------------------------------------------------------
echo "[10/10] Waiting for all deployments ..."

for deploy in prosody jicofo jvb web; do
    until $KUBECTL get deployment "$deploy" -n jitsi 2>/dev/null | grep -q "1/1"; do
        sleep 5
    done
    echo "      jitsi/$deploy: ready"
done

until $KUBECTL get deployment mlflow -n platform 2>/dev/null | grep -q "1/1"; do
    sleep 5
done
echo "      platform/mlflow: ready"

echo ""
echo "============================================"
echo " Deployment complete!"
echo "============================================"
echo ""
$KUBECTL get pods -A -o wide
echo ""
echo " Jitsi         : https://$NIP_DOMAIN"
echo " MLflow        : http://$FLOATING_IP:30500"
echo " MinIO console : http://$FLOATING_IP:30901"
echo " Grafana       : http://$FLOATING_IP:30300 (admin / admin123)"
echo " Prometheus    : http://$FLOATING_IP:30090"
echo " Postgres      : postgres-postgresql.platform.svc.cluster.local:5432 (in-cluster only)"
echo "============================================"
