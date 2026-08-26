from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

from jupyter_client import KernelManager


def execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    runtime = path.parent / ".jupyter_runtime"
    ipython_dir = path.parent / ".ipython_runtime"
    runtime.mkdir(exist_ok=True)
    ipython_dir.mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(ipython_dir)

    manager = KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(path.parent))
    client = manager.client()
    client.start_channels()
    client.wait_for_ready(timeout=60)

    execution_count = 0
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            execution_count += 1
            source = cell.get("source", "")
            print(f"[Notebook] cell {index + 1}/{len(notebook['cells'])}", flush=True)
            message_id = client.execute(source, store_history=True, allow_stdin=False, stop_on_error=True)
            outputs = []
            while True:
                message = client.get_iopub_msg(timeout=3600)
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                kind = message["header"]["msg_type"]
                content = message["content"]
                if kind == "stream":
                    text = content.get("text", "")
                    outputs.append({"output_type": "stream", "name": content.get("name", "stdout"), "text": text})
                    print(text, end="", flush=True)
                elif kind in ("display_data", "execute_result"):
                    data = content.get("data", {})
                    clean_data = {}
                    for mime, value in data.items():
                        if mime == "image/png" and isinstance(value, bytes):
                            value = base64.b64encode(value).decode("ascii")
                        clean_data[mime] = value
                    output = {"output_type": kind, "data": clean_data, "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        output["execution_count"] = content.get("execution_count", execution_count)
                    outputs.append(output)
                elif kind == "error":
                    outputs.append({
                        "output_type": "error",
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    })
                    raise RuntimeError("\n".join(content.get("traceback", [])))
                elif kind == "status" and content.get("execution_state") == "idle":
                    break
            cell["execution_count"] = execution_count
            cell["outputs"] = outputs
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: _execute_notebook.py <notebook.ipynb>")
    execute_notebook(Path(sys.argv[1]).resolve())
