# Plugin system

Capicove plugins are Python distributions named `capicove-<name>`. Discovery uses
Python entry points under `capicove.*` groups rather than filesystem scanning or
implicit imports.

Plugins depend on public contracts and declare compatibility, capabilities,
required resources, side effects, and privacy behavior. Installing a plugin is a
trust decision: plugin code executes with the Python process's permissions unless
the deployment provides stronger isolation.

Public plugin interfaces and entry-point groups require reviewed compatibility
decisions before they change.

## Execution backends

Execution backends use the `capicove.execution_backends` entry-point group. Listing
entry points is side-effect free and does not import plugin modules:

```python
from capicove.plugins import list_execution_backend_plugins

for discovered in list_execution_backend_plugins():
    print(discovered.name, discovered.value)
```

Loading and availability checks remain explicit:

```python
from capicove.plugins import load_execution_backend_plugin

plugin = load_execution_backend_plugin("spark")
availability = plugin.availability()
if availability.available:
    backend = plugin.create({"master": "local[2]"})
```

The loader validates the plugin protocol and contract-version range. Availability
does not create a Spark session; `create()` constructs the selected backend lazily.
