"""YAML topology validation implementing the paper's Figure 2 checks."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ValidationResult:
    """Result of validating a YAML config."""
    valid: bool
    issues: list[str] = field(default_factory=list)


def validate_config(config_path: str) -> ValidationResult:
    """Validate a NASim YAML configuration file.

    Checks:
    - Adjacency matrix is symmetric
    - Subnet count matches topology dimensions
    - Sensitive hosts reference valid (subnet, host) pairs
    - Exploits reference defined services
    - Privilege escalation references defined processes
    """
    issues: list[str] = []
    path = Path(config_path)

    if not path.exists():
        return ValidationResult(valid=False, issues=[f"File not found: {config_path}"])

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        return ValidationResult(valid=False, issues=["Config is not a valid YAML dictionary"])

    subnets = config.get("subnets", [])
    topology = config.get("topology", [])
    sensitive_hosts = config.get("sensitive_hosts", {})
    services = config.get("services", [])
    processes = config.get("processes", [])
    exploits = config.get("exploits", {})
    priv_esc = config.get("privilege_escalation", {})

    # Check topology dimensions
    num_subnets = len(subnets)
    expected_dim = num_subnets + 1  # includes internet node
    if len(topology) != expected_dim:
        issues.append(
            f"Topology has {len(topology)} rows, expected {expected_dim} "
            f"({num_subnets} subnets + 1 internet node)"
        )

    # Check topology symmetry
    for i, row in enumerate(topology):
        if len(row) != len(topology):
            issues.append(f"Topology row {i} has {len(row)} cols, expected {len(topology)}")
        else:
            for j in range(len(row)):
                if j < len(topology) and i < len(topology[j]):
                    if topology[i][j] != topology[j][i]:
                        issues.append(
                            f"Topology not symmetric at ({i},{j}): "
                            f"{topology[i][j]} != {topology[j][i]}"
                        )

    # Check sensitive hosts exist
    # NASim uses 1-based subnet indexing in host keys (subnet 0 = internet node)
    for host_key in sensitive_hosts:
        parsed = _parse_host_key(host_key)
        if parsed is None:
            issues.append(f"Cannot parse sensitive host key: {host_key}")
            continue
        subnet_idx, host_idx = parsed
        if subnet_idx < 1 or subnet_idx > num_subnets:
            issues.append(
                f"Sensitive host {host_key}: subnet {subnet_idx} "
                f"out of range (1-{num_subnets})"
            )
        elif host_idx < 0 or host_idx >= subnets[subnet_idx - 1]:
            issues.append(
                f"Sensitive host {host_key}: host {host_idx} "
                f"out of range for subnet {subnet_idx}"
            )

    # Check exploits reference valid services
    for name, exploit in exploits.items():
        svc = exploit.get("service")
        if svc and svc not in services:
            issues.append(f"Exploit '{name}' references unknown service: {svc}")

    # Check privilege escalation references valid processes
    for name, pe in priv_esc.items():
        proc = pe.get("process")
        if proc and proc not in processes:
            issues.append(f"Privilege escalation '{name}' references unknown process: {proc}")

    return ValidationResult(valid=len(issues) == 0, issues=issues)


def _parse_host_key(key: object) -> tuple[int, int] | None:
    """Parse a host key like '(2, 0)' or (2, 0) into a tuple."""
    if isinstance(key, tuple) and len(key) == 2:
        return key
    if isinstance(key, str):
        cleaned = key.strip("() ")
        parts = cleaned.split(",")
        if len(parts) == 2:
            try:
                return int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                return None
    return None
