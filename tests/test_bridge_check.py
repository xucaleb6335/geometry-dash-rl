from gdrl.env.mock_server import MockServer
from gdrl.tools import bridge_check


def test_mock_server_passes_the_m3_gate_checks(capsys):
    with MockServer(level_seed=2) as server:
        code = bridge_check.main(["--port", str(server.port), "--steps", "200", "--soak", "20"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert out.count("[PASS]") == 3
