"""External socket/DNS guard shared by pytest and Python child processes.

Installed before collection so even import-time probes cannot contact a vendor.
No tests are skipped or excluded. Attempts fail before DNS or socket connection.
"""
import ipaddress
import socket

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_getaddrinfo = socket.getaddrinfo


def local_host(host):
    if host in ("localhost", b"localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def guard_address(address):
    if isinstance(address, tuple) and not local_host(address[0]):
        raise RuntimeError("Offline tests forbid external network access")


def guarded_connect(sock, address):
    guard_address(address)
    return _original_connect(sock, address)


def guarded_connect_ex(sock, address):
    guard_address(address)
    return _original_connect_ex(sock, address)


def guarded_getaddrinfo(host, *args, **kwargs):
    if host is not None and not local_host(host):
        raise RuntimeError("Offline tests forbid external DNS")
    return _original_getaddrinfo(host, *args, **kwargs)


def install():
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo


def restore():
    socket.socket.connect = _original_connect
    socket.socket.connect_ex = _original_connect_ex
    socket.getaddrinfo = _original_getaddrinfo


if __name__ == "sitecustomize":
    install()
