# `capicove jobs`

Plan, queue, observe, cancel, run, and resume durable batch jobs from a version-2
workspace configuration
([RFC-0016](https://github.com/happyBayes/capicove/blob/main/docs/rfcs/0016-operator-jobs-traces-and-models.md)).
See the [operator guide](../guides/operator-jobs.md) for configuration.

```text
capicove jobs plan    --runtime-config PATH --input-root ALIAS --operation OP
                      --provider-fingerprint SHA256 [--parameters JSON] [--output-root ALIAS]
capicove jobs enqueue --runtime-config PATH --input-root ALIAS --operation OP
                      --provider-fingerprint SHA256 [--parameters JSON] --output-root ALIAS
                      --plan-id PLAN_ID --idempotency-key KEY
capicove jobs status  --runtime-config PATH JOB_ID [--offset N] [--limit 1-100]
capicove jobs cancel  --runtime-config PATH JOB_ID --expected-revision N
capicove jobs worker  --config PATH [--once]
capicove jobs resume  --config PATH JOB_ID --expected-revision N
```

`plan`, `enqueue`, `status`, and `cancel` invoke `capicove.batch.*` through the
same host binding as MCP and print the canonical result envelope as JSON. They
exit `0` when the Tool succeeds, `1` when it reports a failure, and `2` when the
configuration cannot start or the arguments are invalid. The configuration must
select the corresponding Tool in `enabled_tools`.

`worker` and `resume` are operator commands. They validate the whole selected
catalog like MCP startup and need `batch:write`. `worker` writes JSON-line events
(`worker_started`, `job_started`, `job_finished`, `job_recovered`, `job_skipped`,
`worker_stopped`) to stderr and exits `0` after a requested stop, `130` after a
second interrupt, and `2` if it cannot start. `resume` prints the requeued job.
