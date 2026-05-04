# Local Oracle 26ai Free

Start:

    onnx2oracle docker up

Start and wait with a bounded SQL probe:

    onnx2oracle docker up --wait --wait-timeout 900

Or manually:

    docker compose up -d

First start is slow (3-5 minutes) while the PDB opens. Subsequent starts are ~30 seconds.

Credentials (for local use only):
- System: `system / onnx2oracle`
- PDB: `FREEPDB1`
- Listener port: `${ORACLE_PORT:-1521}`
- DSN: `system/${ORACLE_PWD:-onnx2oracle}@localhost:${ORACLE_PORT:-1521}/FREEPDB1`

Override the local password before first startup:

    export ORACLE_PWD='your-local-password'

`--target local` uses the same `ORACLE_PWD` value, so CLI loads and verifies keep matching the
container.

If another container already owns port 1521, choose a free host port:

    export ORACLE_PORT=1524

`--target local` uses the same port value.

Run the full real-DB evidence loop from the repo root:

    scripts/run_real_db_integration.sh

It starts Oracle, records the database banner and image id, loads MiniLM through the CLI, verifies
`VECTOR_EMBEDDING`, and runs `pytest tests/test_loader_integration.py --run-integration -v -s`.

## Fallback

If you don't have access to `container-registry.oracle.com` (requires an Oracle SSO login to pull), edit `docker-compose.yml` and replace the image with:

    image: gvenzl/oracle-free:latest

gvenzl's images may lag the latest Oracle release by a few weeks but are auth-free.

You can also switch images without editing YAML:

    ORACLE_IMAGE=container-registry.oracle.com/database/free:latest-lite onnx2oracle docker up --wait
