# Security policy

mlxcat is a local inference server. Its trust model:

* It binds to **loopback** by default (`--host 127.0.0.1`). Exposing it on a
  network interface is your decision; put it behind an authenticated reverse
  proxy if you do.
* It loads model weights from directories you point it at. Malformed
  `config.json` / tokenizer / safetensors files are treated as untrusted input.
* Tool calling and MCP (`/v1/mcp/*`, `--mcp-config`) execute nothing unless you
  configure a server; consent for tool execution belongs to the host app.

## Reporting a vulnerability

Please **do not** open a public issue for security reports. Use GitHub's
private vulnerability reporting on this repository ("Report a vulnerability"
under the Security tab), or email the maintainers via the address on the
[local-ai-cat](https://github.com/local-ai-cat) organization profile.

Include: affected version/commit, a minimal reproduction, and the impact as you
understand it. You will get an acknowledgement within 7 days and a fix or
mitigation plan within 30 days for confirmed issues.

## Supported versions

The `main` branch and the most recent tagged release. Older tags get fixes only
when the fix is trivial to backport.
