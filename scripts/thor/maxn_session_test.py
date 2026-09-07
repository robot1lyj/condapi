import pytest

from scripts.thor.maxn_session import performance_session


class PowerCommands:
    def __init__(self, fail=None):
        self.mode = 1
        self.calls = []
        self.fail = fail

    def __call__(self, *args):
        self.calls.append(args)
        if args == self.fail:
            raise RuntimeError("injected command failure")
        if args[:2] == ("nvpmodel", "-m"):
            self.mode = int(args[2])
        if args == ("nvpmodel", "-q"):
            return f"NV Power Mode: {'MAXN' if self.mode == 0 else '120W'}\n{self.mode}\n"
        return ""


@pytest.mark.parametrize("fail_body", [False, True])
def test_restores_after_success_and_inference_failure(tmp_path, fail_body):
    power = PowerCommands()
    backup = tmp_path / "clocks.conf"
    try:
        with performance_session(backup, run=power):
            assert power.mode == 0
            if fail_body:
                raise ValueError("inference failed")
    except ValueError:
        assert fail_body
    assert power.mode == 1
    assert ("jetson_clocks", "--restore", str(backup)) in power.calls


def test_restores_after_clock_setup_failure(tmp_path):
    power = PowerCommands(fail=("jetson_clocks",))
    with pytest.raises(RuntimeError, match="injected"), performance_session(tmp_path / "clocks.conf", run=power):
        pytest.fail("Inference must not start after setup failure")
    assert power.mode == 1


def test_restores_mode_even_if_clock_restore_fails(tmp_path):
    backup = tmp_path / "clocks.conf"
    power = PowerCommands(fail=("jetson_clocks", "--restore", str(backup)))
    with pytest.raises(RuntimeError, match="injected"), performance_session(backup, run=power):
        pass
    assert power.mode == 1


def test_refuses_to_save_maxn_as_normal_state(tmp_path):
    power = PowerCommands()
    power.mode = 0
    with pytest.raises(RuntimeError, match="normal 120W"), performance_session(tmp_path / "clocks.conf", run=power):
        pytest.fail("Unexpected MAXN starting state")
    assert power.calls == [("nvpmodel", "-q")]
