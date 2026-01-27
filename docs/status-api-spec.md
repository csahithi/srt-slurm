# Status API Specification v1

srtslurm can optionally report job status to an external HTTP API via fire-and-forget POST/PUT requests.

## Configuration

In `srtslurm.yaml` or recipe YAML:

```yaml
reporting:
  status:
    endpoint: "https://status.example.com"
```

If not configured, status reporting is disabled and jobs run normally.

## Endpoints

### POST /api/jobs

Create a job record. Called at submission time.

**Request:**
```json
{
  "job_id": "12345",
  "job_name": "benchmark-run",
  "cluster": "gpu-cluster-01",
  "recipe": "configs/benchmark.yaml",
  "submitted_at": "2025-01-26T10:30:00Z",
  "metadata": {}
}
```

**Response:** `201 Created`

### PUT /api/jobs/{job_id}

Update job status. Called during execution.

**Request:**
```json
{
  "status": "workers",
  "stage": "workers",
  "message": "Starting workers",
  "updated_at": "2025-01-26T10:35:00Z"
}
```

**Response:** `200 OK`

## Status Values

```text
submitted → starting → workers → frontend → benchmark → completed | failed
```

Status reflects which stage is currently executing, not readiness.

## Behavior

- All requests have a 5-second timeout
- Failures are logged at DEBUG level and ignored
- Job execution is never blocked by status reporting failures
