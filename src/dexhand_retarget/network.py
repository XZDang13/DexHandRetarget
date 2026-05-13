from __future__ import annotations

import socket
import ssl
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import TextIO


def get_local_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addresses.add(ip)
    except OSError:
        pass

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        addresses.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()

    addresses.update(get_ifconfig_addresses())

    return sorted(addresses)


def get_ifconfig_addresses() -> list[str]:
    try:
        result = subprocess.run(
            ["ifconfig"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    addresses: set[str] = set()
    current_name = ""
    current_lines: list[str] = []
    for line in result.stdout.splitlines():
        if line and not line.startswith(("\t", " ")):
            _collect_ifconfig_block(current_name, current_lines, addresses)
            current_name = line.split(":", 1)[0]
            current_lines = [line]
        else:
            current_lines.append(line)
    _collect_ifconfig_block(current_name, current_lines, addresses)
    return sorted(addresses)


def _collect_ifconfig_block(name: str, lines: list[str], addresses: set[str]) -> None:
    if not name or name.startswith(("lo", "awdl", "llw", "utun", "gif", "stf")):
        return
    text = "\n".join(lines)
    if "status:" in text and "status: active" not in text:
        return
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "inet":
            ip = parts[1]
            if not ip.startswith(("127.", "169.254.")) and ip != "0.0.0.0":
                addresses.add(ip)


def is_ip_address(value: str) -> bool:
    try:
        socket.inet_pton(socket.AF_INET, value)
        return True
    except OSError:
        pass

    try:
        socket.inet_pton(socket.AF_INET6, value)
        return True
    except OSError:
        return False


def create_self_signed_cert(host: str) -> tuple[tempfile.TemporaryDirectory, Path, Path]:
    temp_dir = tempfile.TemporaryDirectory(prefix="dexhand_retarget_https_")
    temp_path = Path(temp_dir.name)
    cert_path = temp_path / "server.crt"
    key_path = temp_path / "server.key"
    config_path = temp_path / "openssl.cnf"

    names = ["localhost", "127.0.0.1"]
    if host not in {"0.0.0.0", "::"}:
        names.append(host)
    names.extend(get_local_addresses())

    seen: set[str] = set()
    san_entries: list[str] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        prefix = "IP" if is_ip_address(name) else "DNS"
        san_entries.append(f"{prefix}.{len(san_entries) + 1} = {name}")

    config_path.write_text(
        "\n".join(
            [
                "[req]",
                "distinguished_name = req_distinguished_name",
                "x509_extensions = v3_req",
                "prompt = no",
                "",
                "[req_distinguished_name]",
                "CN = DexHandRetarget Local HTTPS",
                "",
                "[v3_req]",
                "subjectAltName = @alt_names",
                "",
                "[alt_names]",
                *san_entries,
                "",
            ]
        ),
        encoding="utf-8",
    )

    command = [
        "openssl",
        "req",
        "-x509",
        "-nodes",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        "7",
        "-config",
        str(config_path),
        "-extensions",
        "v3_req",
        "-sha256",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        temp_dir.cleanup()
        raise RuntimeError("OpenSSL was not found. Install OpenSSL or pass --cert-file and --key-file.") from exc
    except subprocess.CalledProcessError as exc:
        temp_dir.cleanup()
        raise RuntimeError(f"OpenSSL failed to create a self-signed certificate: {exc.stderr}") from exc

    return temp_dir, cert_path, key_path


def create_ssl_context(
    args: Namespace,
    *,
    stream: TextIO | None = None,
) -> tuple[ssl.SSLContext | None, tempfile.TemporaryDirectory | None]:
    stream = stream or sys.stdout
    if not args.https:
        return None, None

    temp_cert_dir = None
    cert_file = args.cert_file
    key_file = args.key_file
    if bool(cert_file) != bool(key_file):
        raise RuntimeError("Use --cert-file and --key-file together.")

    if not cert_file:
        temp_cert_dir, cert_path, key_path = create_self_signed_cert(args.host)
        cert_file = str(cert_path)
        key_file = str(key_path)
        print(f"Generated temporary HTTPS certificate: {cert_file}", file=stream, flush=True)

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    return context, temp_cert_dir


def print_urls(host: str, port: int, scheme: str, *, stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    server_addresses = server_ip_addresses(host)
    print("DexHandRetarget hand receiver", file=stream)
    print(f"Listening on: {scheme}://{host}:{port}", file=stream)
    if len(server_addresses) == 1:
        print(f"Server IP:    {server_addresses[0]}", file=stream)
    elif server_addresses:
        print(f"Server IPs:   {', '.join(server_addresses)}", file=stream)
    else:
        print("Server IP:    not detected", file=stream)
    for address in server_addresses:
        print(f"Client URL:  {scheme}://{address}:{port}", file=stream)
    print("Waiting for WebRTC offer at POST /offer ...", file=stream)


def server_ip_addresses(host: str) -> list[str]:
    if host in {"0.0.0.0", "::"}:
        return get_local_addresses()
    return [host]
