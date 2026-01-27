# Status API Specification v1

srtslurm can optionally report job status to an external HTTP API via fire-and-forget POST/PUT requests.

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `JOB_STATUS_REPORT_ENDPOINT` | Yes | Base URL of the status API |
| `JOB_STATUS_REPORT_CLUSTER` | No | Cluster name for job metadata |

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
  "status": "workers_ready",
  "stage": "workers",
  "message": "All workers healthy",
  "updated_at": "2025-01-26T10:35:00Z"
}
```

**Response:** `200 OK`

## Status Values

```
submitted → starting → head_ready → workers_starting → workers_ready
         → frontend_ready → benchmark → completed | failed
```

## Behavior

- All requests have a 5-second timeout
- Failures are logged at DEBUG level and ignored
- Job execution is never blocked by status reporting failures
