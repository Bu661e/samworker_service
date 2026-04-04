class WorkerIpcError(RuntimeError):
    """Base exception for the worker_ipc package."""


class ProtocolError(WorkerIpcError):
    """Raised when a protocol message is malformed."""


class ClientTimeoutError(WorkerIpcError):
    """Raised when the client times out waiting for a response."""


class WorkerStartError(WorkerIpcError):
    """Raised when a managed child process cannot be started cleanly."""

