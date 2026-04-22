# Local Oracle 26ai Free

Start:

    onnx2oracle docker up

Or manually:

    docker compose up -d

First start is slow (3-5 minutes) while the PDB opens. Subsequent starts are ~30 seconds.

Credentials (for local use only):
- System: `system / onnx2oracle`
- PDB: `FREEPDB1`
- Listener port: `1521`
- DSN: `system/onnx2oracle@localhost:1521/FREEPDB1`

## Fallback

If you don't have access to `container-registry.oracle.com` (requires an Oracle SSO login to pull), edit `docker-compose.yml` and replace the image with:

    image: gvenzl/oracle-free:latest

gvenzl's images may lag the latest Oracle release by a few weeks but are auth-free.
