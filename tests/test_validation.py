import pytest
from cybersim.validation.topology import validate_config, ValidationResult


def test_valid_config_passes():
    result = validate_config("cybersim/configs/baselines/tiny.yaml")
    assert isinstance(result, ValidationResult)
    assert result.valid is True
    assert len(result.issues) == 0


def test_invalid_topology_detected(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("""
subnets: [1, 1]
topology: [[ 1, 1, 0],
           [ 1, 1, 1],
           [ 1, 0, 1]]
sensitive_hosts:
  (1, 0): 100
os: [linux]
services: [ssh]
processes: [tomcat]
exploits:
  e_ssh:
    service: ssh
    os: linux
    prob: 0.8
    cost: 1
    access: user
privilege_escalation:
  pe_tomcat:
    process: tomcat
    os: linux
    prob: 1.0
    cost: 1
    access: root
service_scan_cost: 1
os_scan_cost: 1
subnet_scan_cost: 1
process_scan_cost: 1
host_configurations:
  (1, 0):
    os: linux
    services: [ssh]
    processes: [tomcat]
firewall: {}
step_limit: 1000
""")
    result = validate_config(str(bad_config))
    assert result.valid is False
    assert any("symmetric" in issue.lower() for issue in result.issues)


def test_missing_sensitive_host_detected(tmp_path):
    bad_config = tmp_path / "bad2.yaml"
    bad_config.write_text("""
subnets: [1]
topology: [[ 1, 1],
           [ 1, 1]]
sensitive_hosts:
  (99, 0): 100
os: [linux]
services: [ssh]
processes: [tomcat]
exploits:
  e_ssh:
    service: ssh
    os: linux
    prob: 0.8
    cost: 1
    access: user
privilege_escalation:
  pe_tomcat:
    process: tomcat
    os: linux
    prob: 1.0
    cost: 1
    access: root
service_scan_cost: 1
os_scan_cost: 1
subnet_scan_cost: 1
process_scan_cost: 1
host_configurations: {}
firewall: {}
step_limit: 1000
""")
    result = validate_config(str(bad_config))
    assert result.valid is False
    assert any("sensitive" in issue.lower() for issue in result.issues)
