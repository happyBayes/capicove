"""Identity and deterministic race fixtures preserve the full isolation matrix."""

import importlib
import importlib.util
import threading


def test_identity_matrix_includes_service_and_same_application_users() -> None:
    assert importlib.util.find_spec("tests.remote_support.identities") is not None
    module = importlib.import_module("tests.remote_support.identities")
    subjects = module.principal_matrix()
    assert len({item.tenant for item in subjects}) == 2
    assert all(
        len({item.application for item in subjects if item.tenant == tenant}) == 2
        for tenant in {item.tenant for item in subjects}
    )
    assert len({item.key for item in subjects}) == len(subjects)
    assert {item.kind for item in subjects} == {"human", "service"}
    humans = [
        item
        for item in subjects
        if item.tenant == "tenant-a" and item.application == "app-a" and item.kind == "human"
    ]
    assert len(humans) == 2


def test_fault_barrier_waits_for_explicit_release() -> None:
    assert importlib.util.find_spec("tests.remote_support.barriers") is not None
    module = importlib.import_module("tests.remote_support.barriers")
    barrier = module.FaultBarrier("before-release")
    finished = threading.Event()

    def worker() -> None:
        barrier.arrive()
        finished.set()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        barrier.wait_until_arrived()
        assert not finished.is_set()
    finally:
        barrier.release()
        thread.join(timeout=2)
    assert finished.is_set()
