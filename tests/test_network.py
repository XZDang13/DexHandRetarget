from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dexhand_retarget.network import print_urls  # noqa: E402


class NetworkPrintTests(unittest.TestCase):
    def test_print_urls_shows_detected_server_ips(self) -> None:
        stream = io.StringIO()

        with patch("dexhand_retarget.network.get_local_addresses", return_value=["10.13.11.21"]):
            print_urls("0.0.0.0", 8443, "https", stream=stream)

        output = stream.getvalue()
        self.assertIn("Server IP:    10.13.11.21", output)
        self.assertIn("Client URL:  https://10.13.11.21:8443", output)

    def test_print_urls_shows_explicit_host_as_server_ip(self) -> None:
        stream = io.StringIO()

        print_urls("192.168.1.101", 8443, "https", stream=stream)

        output = stream.getvalue()
        self.assertIn("Server IP:    192.168.1.101", output)
        self.assertIn("Client URL:  https://192.168.1.101:8443", output)


if __name__ == "__main__":
    unittest.main()

