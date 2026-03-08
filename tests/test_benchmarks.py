import pytest
from cybersim.agents.base import EpisodeResult
from cybersim.benchmarks.metrics import compute_metrics, AgentMetrics


def _make_results(n_success: int, n_fail: int) -> list[EpisodeResult]:
    results = []
    for _ in range(n_success):
        results.append(EpisodeResult(compromised=True, steps=10, total_reward=5.0))
    for _ in range(n_fail):
        results.append(EpisodeResult(compromised=False, steps=50, total_reward=-1.0))
    return results


def test_compute_metrics_all_success():
    results = _make_results(10, 0)
    metrics = compute_metrics(results)
    assert isinstance(metrics, AgentMetrics)
    assert metrics.compromise_rate == 1.0
    assert metrics.avg_steps == 10.0


def test_compute_metrics_mixed():
    results = _make_results(7, 3)
    metrics = compute_metrics(results)
    assert metrics.compromise_rate == pytest.approx(0.7)
    assert metrics.avg_steps == pytest.approx(22.0)


def test_compute_metrics_empty():
    metrics = compute_metrics([])
    assert metrics.compromise_rate == 0.0
    assert metrics.avg_steps == 0.0


def test_benchmark_runner_runs():
    from cybersim.benchmarks.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        config_paths=["cybersim/configs/baselines/tiny.yaml"],
        num_episodes=2,
        max_steps=50,
    )
    results = runner.run()
    assert len(results) > 0
    agent_names = {r.agent_name for r in results}
    assert "probing" in agent_names
    assert "killchain" in agent_names
    assert "stealth" in agent_names


def test_generate_plots(tmp_path):
    from cybersim.benchmarks.plots import generate_all_plots

    mock_metrics = [
        AgentMetrics("probing", "tiny", 1.0, 10.0, 2.0, 5.0, 1.0, None, 100),
        AgentMetrics("killchain", "tiny", 0.95, 15.0, 3.0, 4.0, 1.5, None, 100),
        AgentMetrics("stealth", "tiny", 0.8, 28.0, 5.0, 3.0, 2.0, 0.85, 100),
        AgentMetrics("probing", "small", 0.9, 20.0, 4.0, 4.0, 1.0, None, 100),
        AgentMetrics("killchain", "small", 0.85, 25.0, 5.0, 3.5, 1.5, None, 100),
        AgentMetrics("stealth", "small", 0.7, 35.0, 8.0, 2.5, 2.0, 0.78, 100),
    ]

    generate_all_plots(mock_metrics, tmp_path)

    expected_files = [
        "compromise_rates.png",
        "steps_comparison.png",
        "time_comparison.png",
        "scaling.png",
        "agent_radar.png",
        "stealth_tradeoff.png",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists(), f"Missing plot: {filename}"
