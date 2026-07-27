# Cloud Infrastructure & Security Architecture Standard

## 1. Network Topology and Segmentation
- **VPC Isolation**: All microservices and databases must reside within private subnets inside dedicated Virtual Private Clouds (VPCs).
- **Ingress Controls**: Public ingress is strictly restricted to API Gateway proxies behind Cloudflare DDoS protection and Web Application Firewall (WAF) rule sets.
- **Port Restrictions**: SSH (port 22) and direct database connections (ports 5432, 6379, 8000) are disabled across public interfaces. Access is granted exclusively via SSM Session Manager or Zero-Trust Bastion services.

## 2. Container Security and Orchestration
- **Kubernetes Deployment**: Applications run on Managed Kubernetes (EKS) with automated node auto-scaling.
- **Pod Security Admission**: Pods must run as non-root users (`UID 10001`) with read-only root filesystems and privilege escalation explicitly disabled.
- **Base Image Policy**: Docker images must be built from minimal distroless or Alpine base images and undergo daily vulnerability scans via Trivy.

## 3. Storage and Database Governance
- **Vector Database (ChromaDB/Pinecone)**: Vector databases storing semantic embeddings must be hosted in multi-AZ clusters with automated hourly snapshots.
- **Backup Retention**: Production relational databases require point-in-time recovery (PITR) enabled for 35 days, with monthly cross-region backup replication.
