"""
Utilities for running and connecting to a RemoteProcessAllocator over TCP.

Exposes:
- serve_remote_process_allocator: context manager that runs the Rust `process_allocator` TCP server
- create_proc_mesh_remote: helper to create a ProcMesh on one or more remote allocators
"""

# pyre-strict

from __future__ import annotations

import contextlib
import os
import subprocess
from typing import Callable, Generator, Optional, Sequence

from monarch._rust_bindings.monarch_hyperactor.alloc import (
    AllocConstraints,
    AllocSpec,
)
from monarch._rust_bindings.monarch_hyperactor.channel import (
    ChannelAddr,
    ChannelTransport,
)
from monarch._src.actor.allocator import (
    RemoteAllocator,
    StaticRemoteAllocInitializer,
)
from monarch.actor import ProcMesh


@contextlib.contextmanager
def serve_remote_process_allocator(
    *,
    addr: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    program: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> Generator[str, None, None]:
    """
    Start a TCP RemoteProcessAllocator server process and yield its bind address.

    Arguments:
    - addr: Channel address (e.g., "tcp!127.0.0.1:26600"). If None, bind to an ephemeral TCP port.
    - timeout_sec: Optional inactivity timeout; server exits if no allocations occur before timeout.
    - program: Optional bootstrap program for spawned processes (defaults to process allocator's default).
    - env: Extra environment vars for the server process.

    Yields:
    - The channel address string the server is bound to.
    """

    bind_addr = addr or ChannelAddr.any(ChannelTransport.Tcp)

    args: list[str] = [
        "process_allocator",
        f"--addr={bind_addr}",
    ]
    if program is not None:
        args.append(f"--program={program}")
    if timeout_sec is not None:
        args.append(f"--timeout-sec={timeout_sec}")

    popen_env = os.environ.copy()
    if env:
        popen_env.update(env)

    proc = subprocess.Popen(
        args=args,
        env=popen_env,
    )
    try:
        yield bind_addr
    finally:
        proc.terminate()
        try:
            # 5 seconds should be sufficient for graceful shutdown
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def create_proc_mesh_remote(
    *,
    world_id: str,
    hosts: Sequence[str],
    spec: AllocSpec,
    setup: Optional[Callable[[], None]] = None,
) -> ProcMesh:
    """
    Create a ProcMesh using remote process allocators running at the provided host addresses.

    Arguments:
    - world_id: Identifier for the process world.
    - hosts: Sequence of channel addresses (e.g., ["tcp!127.0.0.1:26600", ...]).
    - spec: AllocSpec describing the mesh dimensions (e.g., host=2, gpu=4).
    - setup: Optional function to run in each allocated process before actor startup.

    Returns:
    - A ProcMesh instance. Use `await proc_mesh.initialized` before spawning actors.
    """

    initializer = StaticRemoteAllocInitializer(*hosts)
    allocator = RemoteAllocator(world_id=world_id, initializer=initializer)
    alloc = allocator.allocate(spec)
    return ProcMesh.from_alloc(alloc, setup=setup)


def alloc_spec(*, constraints: Optional[dict[str, str]] = None, **dims: int) -> AllocSpec:
    """
    Convenience constructor for AllocSpec.

    Example:
        spec = alloc_spec(host=2, gpu=4)
    """

    return AllocSpec(AllocConstraints(match_labels=constraints or {}), **dims)

