"""Pytest configuration for the sandbox test suite.

The production sandbox requires bearer-token authentication on the
`/api/v1/execute` endpoints. To keep the test suite self-contained we
enable the development bypass (SANDBOX_AUTH_DISABLED=true) by default.

For tests that explicitly want to verify the authentication logic, set
SANDBOX_AUTH_TOKEN and unset SANDBOX_AUTH_DISABLED in the test body.
"""

import os

# Default: enable dev bypass so existing tests don't break
os.environ.setdefault("SANDBOX_AUTH_DISABLED", "true")